import sys
from pathlib import Path

CURRENT_PATH = Path(__file__).resolve().parent
if str(CURRENT_PATH) not in sys.path:
    sys.path.insert(0, str(CURRENT_PATH))

from omagent_core.utils.container import container
from omagent_core.utils.registry import registry

# Register workers by importing the workers directory
registry.import_module(project_path=CURRENT_PATH.joinpath("workers"))

container.register_callback(callback="AppCallback")
container.register_input(input="AppInput")
container.register_stm("SharedMemSTM")

# Compile container config
container.compile_config(CURRENT_PATH)
print("Successfully compiled container configuration into container.yaml")
