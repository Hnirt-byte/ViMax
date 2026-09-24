import logging
import os
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from interfaces import CharacterInScene, ImageOutput
from langchain_core.messages import HumanMessage, SystemMessage



prompt_template_front = \
"""
Generate a full-body, front-view portrait of character {identifier} based on the following description, with a pure white background. Use {canvas_description}. The character should be centered in the image. Gazing straight ahead. Standing with arms relaxed at sides. Natural expression.
Features: {features}
Style: {style}
"""

prompt_template_side = \
"""
Generate a full-body, side-view portrait of character {identifier} based on the provided front-view portrait, with a pure white background. Use {canvas_description}. The character should be centered in the image. Facing left. Standing with arms relaxed at sides.
"""

prompt_template_back = \
"""
Generate a full-body, back-view portrait of character {identifier} based on the provided front-view portrait, with a pure white background. Use {canvas_description}. The character should be centered in the image. No facial features should be visible.
"""


class CharacterPortraitsGenerator:
    def __init__(
        self,
        image_generator,
    ):
        self.image_generator = image_generator


    async def generate_front_portrait(
        self,
        character: CharacterInScene,
        style: str,
        image_size: str | None = None,
    ) -> ImageOutput:
        features = "(static) " + (character.static_features or "") + "; (dynamic) " + (character.dynamic_features or "")
        prompt = prompt_template_front.format(
            identifier=character.identifier_in_scene,
            features=features,
            style=style,
            canvas_description=_canvas_description(image_size),
        )
        kwargs = {"prompt": prompt}
        if image_size:
            kwargs["size"] = image_size
        image_output = await self.image_generator.generate_single_image(**kwargs)
        return image_output

    async def generate_side_portrait(
        self,
        character: CharacterInScene,
        front_image_path: str,
        image_size: str | None = None,
    ) -> ImageOutput:
        prompt = prompt_template_side.format(
            identifier=character.identifier_in_scene,
            canvas_description=_canvas_description(image_size),
        )
        kwargs = {"prompt": prompt, "reference_image_paths": [front_image_path]}
        if image_size:
            kwargs["size"] = image_size
        image_output = await self.image_generator.generate_single_image(**kwargs)
        return image_output


    async def generate_back_portrait(
        self,
        character: CharacterInScene,
        front_image_path: str,
        image_size: str | None = None,
    ) -> ImageOutput:
        prompt = prompt_template_back.format(
            identifier=character.identifier_in_scene,
            canvas_description=_canvas_description(image_size),
        )
        kwargs = {"prompt": prompt, "reference_image_paths": [front_image_path]}
        if image_size:
            kwargs["size"] = image_size
        image_output = await self.image_generator.generate_single_image(**kwargs)
        return image_output


def _canvas_description(image_size: str | None) -> str:
    if not image_size:
        return "a canvas appropriate to the requested composition"
    return f"a {image_size} canvas matching the requested output orientation"