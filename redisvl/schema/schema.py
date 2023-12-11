import yaml

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel, ValidationError

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


def create_vector_field(data: Dict[str, Any]) -> Union[FlatVectorField, HNSWVectorField]:
    vector_field_classes = {
        'flat': FlatVectorField,
        'hnsw': HNSWVectorField
    }
    algorithm = data.get('algorithm', '').lower()
    field_class = vector_field_classes.get(algorithm)
    if field_class is None:
        raise ValueError(f"Unknown vector field algorithm: {algorithm}")
    return field_class(**data)


class Schema(BaseModel):
    name: str
    prefix: str = "rvl"
    key_separator: str = ":"
    storage_type: StorageType = StorageType.HASH
    fields: Dict[str, List[BaseField]] = {}

    _FIELD_TYPE_MAP = {
        "tag": TagField,
        "text": TextField,
        "numeric": NumericField,
        "geo": GeoField,
        "vector": create_vector_field
    }

    @classmethod
    def from_yaml(cls, file_path: str):
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
            yaml_data = yaml.safe_load(f)

        return cls.from_dict(yaml_data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        schema = cls(**data['index'])
        for field_type, field_list in data['fields'].items():
            for field_data in field_list:
                # make use of our add field method!
                schema.add_field(field_type, **field_data)
        return schema

    def _get_field_class(self, field_type: str) -> Type[BaseField]:
        return self._FIELD_TYPE_MAP.get(field_type)

    def add_field(self, field_type: str, **kwargs):
        if field_type == 'vector':
            # Ensure a vector field of the same name isn't already added
            existing_fields = self.fields.get(field_type, [])
            if any(field.name == kwargs.get('name') for field in existing_fields):
                raise ValueError(f"Field with name '{kwargs.get('name')}' already exists in vector fields.")

            field = create_vector_field(kwargs)
        else:
            field_class = self._get_field_class(field_type)
            try:
                field = field_class(**kwargs)
            except ValidationError as e:
                raise ValueError(f"Error adding field: {e}")

            # Ensure a field of the same name isn't already added
            existing_fields = self.fields.get(field_type, [])
            if any(field.name == kwargs.get('name') for field in existing_fields):
                raise ValueError(f"Field with name '{kwargs.get('name')}' already exists in {field_type} fields.")

        self.fields.setdefault(field_type, []).append(field)

    def remove_field(self, field_type: str, field_name: str):
        if field_type in self.fields:
            self.fields[field_type] = [
                field for field in self.fields[field_type] if field.name != field_name]

    @property
    def index_fields(self) -> list:
        redis_fields = []
        for field_name in self.fields:
            if field_type := getattr(self.fields, field_name):
                for field in field_type:
                    redis_fields.append(field.as_field())
        return redis_fields

    def _test_numeric(self, value) -> bool:
        """Test if a value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _infer_type(self, value) -> Optional[str]:
        """Infer the type of a value."""
        if value in [None, ""]:
            return None
        if self._test_numeric(value):
            return "numeric"
        if isinstance(value, (list, set, tuple)) and all(
            isinstance(v, str) for v in value
        ):
            return "tag"
        if isinstance(value, str):
            return "text"
        # Check if JSON or HASH
        if self.storage_type == StorageType.JSON and isinstance(value, list) and all(
            self._test_numeric(v) for v in value
        ):
            # return vector field
            return "vector"
        if self.storage_type == StorageType.JSON and isinstance(value, bytes):
            # return vector field
            return "vector"
        # TODO - how to determine Flat or HNSW?
        # TODO - do we convert the vector to a bytes array if it's a list for hash structure???
        # TODO - geo????


    def generate_fields(
        self,
        data: Dict[str, Any],
        strict: bool = False,
        ignore_fields: List[str] = [],
        field_args: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, List[Dict[str, Any]]]:
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
        fields: Dict[str, List[Dict[str, Any]]] = {}

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

            field_class = self._get_field_class(field_type)
            fields.setdefault(field_type, []).append(field_class(**field_kwargs))

        return fields

    def to_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Dump the RedisVL schema to a dictionary.

        Returns:
            The RedisVL schema as a dictionary.
        """
        index_data = {
            'name': self.name,
            'prefix': self.prefix,
            'key_separator': self.key_separator,
            'storage_type': self.storage_type.value
        }
        formatted_fields = {}
        for field_type, fields in self.fields.items():
            formatted_fields[field_type] = [field.dict(exclude_unset=True) for field in fields]
        return {'index': index_data, 'fields': formatted_fields}

    def to_yaml(self, file_path: str) -> None:
        """
        Write the schema to a yaml file.

        Args:
            file_path (str): The yaml file path where the RedisVL schema is written.

        Raises:
            TypeError: If the provided file path is not a valid YAML file.
        """
        if not file_path.endswith(".yaml"):
            raise TypeError("Invalid file path. Must be a YAML file.")

        schema = self.to_dict()
        with open(file_path, "w") as f:
            yaml.dump(schema, sort_keys=False)


