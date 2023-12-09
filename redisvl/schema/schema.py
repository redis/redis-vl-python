import yaml

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel

from redisvl.schema.fields import (
    BaseField,
    BaseVectorField,
    TagField,
    TextField,
    NumericField,
    FlatVectorField,
    HNSWVectorField,
    GeoField
)



class StorageType(Enum):
    HASH = "hash"
    JSON = "json"


class IndexModel(BaseModel):
    name: str
    prefix: str = "rvl"
    key_separator: str = ":"
    storage_type: StorageType = StorageType.HASH


class FieldsModel(BaseModel):
    tag: List[TagField] = []
    text: List[TextField] = []
    numeric: List[NumericField] = []
    geo: List[GeoField] = []
    vector: List[Union[FlatVectorField, HNSWVectorField]] = []

    _FIELD_TYPE_MAP = {
        TagField: 'tag',
        TextField: 'text',
        NumericField: 'numeric',
        GeoField: 'geo',
        BaseVectorField: 'vector'
    }

    def add(self, field: Union[BaseField, BaseVectorField]):
        """
        Add a field to the schema. The field is appended to the list corresponding to its type.

        Args:
            field (Union[BaseField, BaseVectorField]): An instance of a field type to be added.

        Raises:
            TypeError: If the field type is not supported.
        """
        field_list_name = self._FIELD_TYPE_MAP.get(type(field))
        if field_list_name:
            getattr(self, field_list_name).append(field)
        else:
            raise TypeError(f"Invalid field type: {type(field).__name__}. Expected one of TagField, TextField, NumericField, GeoField, or BaseVectorField.")
        return self

    def remove(self, field_type: Type[Union[BaseField, BaseVectorField]], field_name: str):
        """
        Remove a field from the schema based on its type and name.

        Args:
            field_type (Type[Union[BaseField, BaseVectorField]]): The type of
                the field to remove.
            field_name (str): The name of the field to remove.

        Raises:
           ValueError: If the field type is not found in the model.
        """
        field_list_name = self._FIELD_TYPE_MAP.get(field_type)
        if field_list_name:
            field_list = getattr(self, field_list_name)
            field_list[:] = [field for field in field_list if field.name != field_name]
        else:
            raise ValueError(f"No field list found for type {field_type.__name__}")
        return self


class Schema(BaseModel):

    index: IndexModel
    fields: FieldsModel = FieldsModel()

    @property
    def index_name(self) -> str:
        return self.index.name

    @property
    def index_prefix(self) -> str:
        return self.index.prefix

    @property
    def key_separator(self) -> str:
        return self.index.key_separator

    @property
    def storage_type(self) -> StorageType:
        return self.index.storage_type

    @property
    def index_fields(self) -> list:
        redis_fields = []
        for field_name in self.fields.__fields__.keys():
            field_type = getattr(self.fields, field_name)
            if field_type:
                for field in field_type:
                    redis_fields.append(field.as_field())
        return redis_fields

    @classmethod
    def from_params(
        cls,
        name: str,
        prefix: str = "rvl",
        key_separator: str = ":",
        storage_type: StorageType = StorageType.HASH,
        fields: Dict[str, List[Dict[str, Any]]] = {},
    ):
        """
        Create a Schema instance from provided parameters.

        Returns:
            A Schema instance.
        """
        # TODO how else should we shape this constructor???
        index = IndexModel(
            name=name,
            prefix=prefix,
            key_separator=key_separator,
            storage_type=storage_type
        )
        fields = FieldsModel(**fields)
        return cls(**index, fields=fields)

    def _test_numeric(self, value) -> bool:
        """Test if a value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _infer_type(self, value) -> Optional[Union[BaseField, BaseVectorField]]:
        """Infer the type of a value."""
        if value in [None, ""]:
            return None
        if self._test_numeric(value):
            return NumericField
        if isinstance(value, (list, set, tuple)) and all(
            isinstance(v, str) for v in value
        ):
            return TagField
        if isinstance(value, str):
            return TextField
        # Check if JSON or HASH
        if self.storage_type == StorageType.JSON and isinstance(value, list) and all(
            self._test_numeric(v) for v in value
        ):
            # return vector field
            pass
        if self.storage_type == StorageType.JSON and isinstance(value, bytes):
            # return vector field
            pass
        # TODO - how to determine Flat or HNSW?
        # TODO - do we convert the vector to a bytes array if it's a list for hash structure???
        # TODO - geo????


    def generate_fields(
        self,
        data: Dict[str, Any],
        strict: bool = False,
        ignore_fields: List[str] = [],
        field_args: Dict[str, Dict[str, Any]] = {}
    ) -> FieldsModel:
        """
        Generate a new FieldsModel from sample data, with options for strict
        type checking, ignoring specific fields, and additional field arguments.

        Args:
            data (Dict[str, Any]): Sample data used to infer field types.
            strict (bool, optional): If True, raises an error when a field type 
                can't be inferred. Defaults to False.
            ignore_fields (List[str], optional): List of field names to ignore.
                Defaults to [].
            field_args (Dict[str, Dict[str, Any]], optional): Additional
                arguments for each field. Defaults to {}.

        Raises:
            ValueError: If strict is True and a field type cannot be inferred.

        Returns:
            FieldsModel: A new FieldsModel instance populated with inferred
                fields.
        """
        fields = FieldsModel()

        for field_name, value in data.items():
            # skip field if ignored
            if field_name in ignore_fields:
                continue

            field_type = self._infer_type(value)
            if not field_type:
                warning_message = f"Warning: Unable to determine field type for '{field_name}' with value '{value}'"
                if strict:
                    raise ValueError(warning_message)
                print(warning_message)
                continue

            # build field kwargs from defaults and then overriding with field_args supplied into function
            field_kwargs = {"name": field_name}
            field_kwargs.update(field_args.get(field_name, {}))

            if self.storage_type == StorageType.JSON:
                # make JSON specific schema modifications
                field_kwargs["as_name"] = field_name
                field_kwargs["name"] = f"$.{field_name.replace(' ', '')}"

            fields.add(field_type(**field_kwargs))

        return fields

    def to_dict(self) -> Dict[str, Any]:
        """
        Dump the RedisVL schema to a dictionary.

        Returns:
            The RedisVL schema as a dictionary.
        """
        schema = self.dict()
        return schema

    def to_yaml(self, path: str) -> None:
        """
        Write the schema to a yaml file.

        Args:
            path (str): The yaml file path where the RedisVL schema is written.

        Raises:
            TypeError: If the provided file path is not a valid YAML file.
        """
        if not path.endswith(".yaml"):
            raise TypeError("Invalid file path. Must be a YAML file.")

        schema = self.dump()
        with open(path, "w") as f:
            yaml.dump(schema, f)


def read_schema(file_path: str) -> Schema:
    """
    Create a Schema instance from a YAML file.
    Args:
        file_path: The path to the YAML file.
    Returns:
        A Schema instance.
    Raises:
        ValueError: If the file path is not a YAML file.
        FileNotFoundError: If the YAML file does not exist.
    """
    if not file_path.endswith(".yaml"):
        raise ValueError("Must provide a valid YAML file path")

    fp = Path(file_path).resolve()
    if not fp.exists():
        raise FileNotFoundError(f"Schema file {file_path} does not exist")

    with open(fp, "r") as f:
        schema = yaml.safe_load(f)

    return Schema(**schema)

