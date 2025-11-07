# simuglue/cli/_completers.py
try:
    from argcomplete.completers import (
        FilesCompleter,
        ChoicesCompleter,
        DirectoriesCompleter,
        EnvironCompleter,
    )
except ImportError:  # fallback stubs
    FilesCompleter = ChoicesCompleter = DirectoriesCompleter = EnvironCompleter = None
