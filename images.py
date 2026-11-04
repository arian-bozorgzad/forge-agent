"""Validate one small inline image. No URL fetching or filesystem uploads YET ."""

import base64
import binascii
import io
import warnings

from PIL import Image
from pydantic import Field, field_validator

from app.schemas import StrictModel


class ImageInput(StrictModel):
    data_url: str = Field(max_length=44000)

    @field_validator("data_url")
    @classmethod
    def valid_image(cls, value):
        try:
            header, encoded = value.split(",", 1)
            formats = {"data:image/png;base64": "PNG", "data:image/jpeg;base64": "JPEG"}
            if header not in formats:
                raise ValueError("Use a PNG or JPEG image")
            raw = base64.b64decode(encoded, validate=True)
            if not raw or len(raw) > 32768:
                raise ValueError("Image must be at most 32 KiB after resizing")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != formats[header] or max(image.size) > 1600:
                        raise ValueError("Image format mismatch or dimensions exceed 1600 pixels")
                    image.verify()
        except (
            OSError,
            binascii.Error,
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
        ) as error:
            raise ValueError("Invalid image") from error
        return value
