"""Configuration of project"""

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class PreprocessingConfig(BaseModel):
    """Config for preprocessing"""

    target_sr: int = 16000
    n_fft: int = 512
    hop_len: int = 160


class Config(BaseSettings):
    """Main config"""

    model_config = SettingsConfigDict(
        yaml_file="config.yaml", yaml_file_encoding="utf-8"
    )
    preprocessing: PreprocessingConfig = PreprocessingConfig()

    # pylint: disable=R0913,R0917
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


config = Config()
