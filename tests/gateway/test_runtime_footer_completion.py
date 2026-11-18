"""A footered reply must survive terminal publication and reconnect replay."""
import asyncio
from copy import copy, deepcopy

import pytest

from gateway.runtime_footer import FooteredReply
from gateway.session_events import SessionEvents
from hermes_state_runtime import get_session_admission
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


@pytest.mark.parametrize('body,footer', [('result', 'model · 42%'), ('', ''),
                                        ('result\n\nfooter', 'footer'), ('日本語 🧪', '工程')])
def test_footer_copy_publishes_and_replays_without_duplication(body, footer):
    reply = FooteredReply(body, footer)
    stream = SessionEvents()
    received = []
    stream.observers.add(received.append)
    reply.extra = {'items': []}
    cloned = deepcopy(reply)
    cloned.extra['items'].append('changed')
    assert reply.extra == {'items': []}
    assert copy(reply).runtime_footer == footer
    assert stream.publish('session', {'text': reply, 'content': reply})
    assert received[0]['params']['payload']['text'] == body + '\n\n' + footer
    replay = stream.since(stream.epoch, 0)
    assert replay['count'] == 1
    assert replay['events'][0]['payload']['text'] == str(reply)
    assert stream.since(stream.epoch, replay['latest_seq'])['events'] == []


@pytest.mark.asyncio
async def test_footer_settles_publishes_and_same_request_never_reexecutes(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    executed, received = [], []
    reply = FooteredReply('verified work', 'runtime metadata')

    async def execute(owner, ref, row):
        executed.append(row['admission_id'])
        owner.pending_results[row['admission_id']] = {'result': {'final_response': reply}, 'usage': {}}
        return reply

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    stream = authority.sessions['s'].event_stream
    stream.observers.add(received.append)
    with db:
        receipt = await _submit(authority, 'same-work')
        await asyncio.wait_for(authority._drain(REF), 10)
        row = get_session_admission(db, admission_id=receipt.admission_id)
        assert (row['status'], row['outcome']) == ('terminal', 'completed')
        completions = [f for f in received if f['params']['type'] == 'message.complete']
        assert len(completions) == 1
        assert completions[0]['params']['payload']['text'] == str(reply)
        replay = stream.since(stream.epoch, 0)
        assert len([f for f in replay['events'] if f['type'] == 'message.complete']) == 1
        duplicate = await _submit(authority, 'same-work')
        assert duplicate.admission_id == receipt.admission_id
        await asyncio.wait_for(authority._drain(REF), 10)
        assert executed == [receipt.admission_id]
        assert len([f for f in received if f['params']['type'] == 'message.complete']) == 1
