"""Opt-in MongoDB test. Requires a dedicated disposable test server, never production."""

import os
import uuid

import pytest

from src.repositories import MongoAuditRepository
from src.schemas import CallAuditRecord
from tests.test_pipeline import document


@pytest.mark.integration
def test_real_mongodb_replay_and_indexes():
    uri = os.getenv("MONGODB_TEST_URI")
    if not uri:
        pytest.skip("set MONGODB_TEST_URI to a disposable MongoDB instance")
    from pymongo import MongoClient

    name = "altur_test_" + uuid.uuid4().hex
    with MongoClient(uri, timeoutMS=5000, tz_aware=True) as client:
        try:
            collection = client[name]["events"]
            repository = MongoAuditRepository(collection)
            repository.ensure_indexes()
            record = CallAuditRecord.model_validate(document())
            assert repository.save(record)
            assert not repository.save(record)
            assert collection.count_documents({}) == 1
            assert collection.find_one()["timestamp"].tzinfo is not None
        finally:
            client.drop_database(name)
