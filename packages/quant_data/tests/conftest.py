import pytest
from quant_data.db.models import Base
from quant_data.db.repositories import MarketDataRepository
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = TestingSessionLocal()
    yield session
    session.close()


@pytest.fixture
def repo(db_session):
    return MarketDataRepository(db_session)
