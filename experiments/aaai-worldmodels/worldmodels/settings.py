"""Resolve project paths and dotenv without displaying secrets."""
from pathlib import Path
from dotenv import load_dotenv as load_env

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parents[1]


def find_env():
    """Only use this experiment's .env, never an unrelated parent file."""
    return str(PROJECT / '.env')


def configure():
    load_env(find_env())
    return PROJECT, REPOSITORY
