"""Compatibility entry point; never inserts demo data into a real database."""

from src.export_training import main

if __name__ == "__main__":
    main()
