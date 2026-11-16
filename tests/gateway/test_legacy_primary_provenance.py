"""Standalone native bindings survive multiplex enablement, never a change of owner.

Exercise the real producer injection, durable FIFO and consumer with only model/transport
I/O replaced. No production bindings or failed receipts are replayed.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_wake_during_restart_drain_authority import authority_route, start_human_turn
from tests.waiting import async_wait_until


def enable_multiplex(runner, authority, monkeypatch):
    home = Path(authority.db.db_path).resolve().parent
    monkeypatch.setattr('hermes_constants.get_process_hermes_home', lambda: home)
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = [('default', home)]
    return home


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_history', [False, True])
async def test_old_primary_completion_consumes_once_without_rewriting_history(monkeypatch, failed_history):
    executed = []
    runner, adapter, authority, source = await authority_route(monkeypatch, executed)
    try:
        await start_human_turn(runner, adapter, authority, source, None)
        route = runner.session_store._generate_session_key(source)
        sid = runner.session_store.peek_session_id(route)
        if failed_history:
            from gateway.platforms.event import MessageEvent
            original_turn = runner._handle_message_with_agent

            async def fail(*args, **kwargs):
                raise RuntimeError('fixture consumer failure')

            monkeypatch.setattr(runner, '_handle_message_with_agent', fail)
            await adapter.handle_message(MessageEvent(text='failed-before-switch', source=source, message_id='failed'))
            await async_wait_until(lambda: len(list_session_admissions(
                authority.db, session_id=sid, pending_only=False)) == 2 and all(
                    row['status'] == 'terminal' for row in list_session_admissions(
                        authority.db, session_id=sid, pending_only=False)), what='failed consumer receipt')
            assert list_session_admissions(authority.db, session_id=sid, pending_only=False)[-1]['outcome'] == 'failed'
            monkeypatch.setattr(runner, '_handle_message_with_agent', original_turn)
        before = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert 'transport_profile' not in before[0]['payload']['native_text_v1']['provenance']
        enable_multiplex(runner, authority, monkeypatch)
        event = dict(type='completion', session_id='fresh-bounded-process', started_at=1.0,
                     session_key=route, parent_session_id=sid)
        assert await runner._inject_watch_notification('fresh-result', event) is True
        await async_wait_until(lambda: len(executed) == 2 and all(
            r['status'] == 'terminal' for r in list_session_admissions(
                authority.db, session_id=sid, pending_only=False)), what='completion consumer terminal receipt')
        after = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert after[:-1] == before
        assert after[-1]['outcome'] == 'completed'
        assert 'fresh-result' in executed[-1]
        assert await runner._inject_watch_notification('fresh-result', event) is True
        assert list_session_admissions(authority.db, session_id=sid, pending_only=False) == after
        assert len(executed) == 2
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['current', 'wrong-profile', 'wrong-runtime', 'wrong-transport',
                                 'wrong-connector', 'missing-connector', 'missing-home',
                                 'ambiguous-home', 'ambiguous-adapter', 'ambiguous-route'])
async def test_legacy_binding_does_not_relax_identity_checks(monkeypatch, case):
    from gateway.config import Platform
    from gateway.profile_routing import ProfileRoute
    from gateway.session import SessionSource
    from gateway.session_ingress_context import restore_provenance
    executed = []
    runner, adapter, authority, source = await authority_route(monkeypatch, executed)
    try:
        await start_human_turn(runner, adapter, authority, source, None)
        sid = runner.session_store.peek_session_id(runner.session_store._generate_session_key(source))
        before = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        envelope = before[0]['payload']['native_text_v1']
        provenance = deepcopy(envelope['provenance'])
        restored = SessionSource.from_dict(envelope['source'])
        home = enable_multiplex(runner, authority, monkeypatch)
        if case == 'current':
            provenance['transport_profile'] = None
            assert restore_provenance(runner, restored, provenance) == home
        else:
            if case == 'wrong-profile':
                restored.profile = 'foreign'
            elif case == 'wrong-runtime':
                provenance['runtime_home'] = str(home / 'foreign')
            elif case == 'wrong-transport':
                provenance['transport_home'] = str(home / 'foreign')
            elif case == 'wrong-connector':
                provenance['connector'] = 'changed'
            elif case == 'missing-connector':
                del provenance['connector']
            elif case == 'missing-home':
                runner.config._runtime_profile_homes = []
            elif case == 'ambiguous-home':
                runner.config._runtime_profile_homes.append(('alias', home))
            elif case == 'ambiguous-adapter':
                runner._profile_adapters = {'alias': {Platform.DISCORD: adapter}}
            elif case == 'ambiguous-route':
                route = ProfileRoute(name='a', platform='discord', profile='default', chat_id='42')
                runner.config.profile_routes = [route, route]
            with pytest.raises(RuntimeStoreError):
                restore_provenance(runner, restored, provenance)
        assert list_session_admissions(authority.db, session_id=sid, pending_only=False) == before
        assert executed == ['A']
    finally:
        authority.db.close()
