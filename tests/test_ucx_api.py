import pickle

from ucp._libs import ucx_api


def test_pickle_ucx_address():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.get_address()
    dumped_address = pickle.dumps(org_address)
    org_address_hash = hash(org_address)
    org_address = bytes(org_address)
    new_address = pickle.loads(dumped_address)
    assert org_address_hash == hash(new_address)
    assert bytes(org_address) == bytes(new_address)
