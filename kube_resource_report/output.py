import logging
import shutil
from pathlib import Path
from typing import Optional

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from kube_resource_report import filters

TEMPLATES_PATH = Path(__file__).parent / "templates"

logger = logging.getLogger(__name__)


class OutputManager:
    def __init__(
        self,
        output_path: Path,
        templates_path: Optional[Path] = None,
        prerender_hook=None,
    ):
        self.output_path = output_path
        self.written_paths: set = set()

        templates_paths = [str(TEMPLATES_PATH)]
        if templates_path:
            # prepend the custom template path so custom templates will overwrite any default ones
            templates_paths.insert(0, str(templates_path))

        self.prerender_hook = prerender_hook

        env = Environment(
            loader=FileSystemLoader(templates_paths),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.filters["money"] = filters.money
        env.filters["cpu"] = filters.cpu
        env.filters["memory"] = filters.memory
        self.env = env

    def open(self, file_name, mode="w"):
        path = self.output_path / file_name
        self.written_paths.add(path)
        logger.info(f"Writing {file_name}..")
        return path.open(mode)

    def exists(self, file_name: str):
        path = self.output_path / file_name
        return path.exists()

    def copy_static_assets(self):
        assets_path = self.output_path / "assets"
        assets_path.mkdir(exist_ok=True)

        assets_source_path = TEMPLATES_PATH / "assets"

        for path in assets_source_path.iterdir():
            if path.match("*.js") or path.match("*.css") or path.match("*.png"):
                destination_path = assets_path / path.name
                self.written_paths.add(destination_path)
                shutil.copy(str(path), str(destination_path))

    def render_template(self, template_name: str, context: dict, output_file_name: str):
        path = self.output_path / output_file_name
        self.written_paths.add(path)

        if self.prerender_hook:
            self.prerender_hook(template_name, context)

        logger.info(f"Generating {output_file_name}..")
        template = self.env.get_template(template_name)
        template.stream(**context).dump(str(path))

    def clean_up_stale_files(self):
        for path in self.output_path.iterdir():
            if path.is_file() and path not in self.written_paths:
                logger.info(f"Cleaning up {path.name}..")
                path.unlink()
