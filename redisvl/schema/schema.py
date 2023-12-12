import re
import yaml
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple, Optional

from pydantic import BaseModel, ValidationError

from redisvl.schema.fields import (
    BaseField,
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


def get_vector_type(**field_data: Dict[str, Any]) -> Union[FlatVectorField, HNSWVectorField]:
    """Get the vector field type from the field data."""

    vector_field_classes = {
        'flat': FlatVectorField,
        'hnsw': HNSWVectorField
    }
    algorithm = field_data.get('algorithm', '').lower()
    if algorithm not in vector_field_classes.keys():
        raise ValueError(f"Unknown vector field algorithm: {algorithm}")

    # default to FLAT
    return vector_field_classes.get(algorithm, FlatVectorField)(**field_data)


class IndexSchema(BaseModel):
    """
    RedisVL index schema for stroring and indexing vectors and metadata
    fields in Redis.

    Attributes:
        name (str): The name of the index.
        prefix (str): The key prefix used in the Redis database keys.
        key_separator (str): The key separator used in the Redis database keys.
        storage_type (StorageType): The Redis storage type for underlying data.
        fields (Dict[str, List[BaseField]]): The defined index fields.
    """
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
        "vector": get_vector_type
    }

    @property
    def index_fields(self) -> list:
        """Returns a list of index fields in the Redis database."""
        redis_fields = []
        for field_list in self.fields.values():
            redis_fields.extend(field.as_field() for field in field_list)
        return redis_fields

    def add_fields(self, fields: Dict[str, List[Dict[str, Any]]]):
        """Adds multiple fields to the index schema."""
        for field_type, field_list in fields.items():
            for field_data in field_list:
                self.add_field(field_type, **field_data)

    def _create_field_instance(
        self,
        field_name: str,
        value: Optional[Any] = None,
        field_type: Optional[str] = None,
        field_args: Dict[str, Dict[str, Any]] = {}
    ) -> Tuple[str, BaseField]:
        """
        Creates an instance of a field. This method can create a field instance
        based on either a specified field type or by inferring the type from a
        provided value.

        Args:
            field_name (str): The name of the field.
            value (Optional[Any], optional): The value used for type inference.
                Optional if field_type is specified. Defaults to None.
            field_type (Optional[str], optional): The type of the field. Optional
                if value is provided for type inference. Defaults to None.
            field_args: Additional arguments for the field creation.

        Returns:
            A tuple containing the field type and the field instance.

        Raises:
            ValueError: If neither value nor field_type is provided, or if the
                field type is unknown or non-inferrable.
        """
        if field_type is None and value is not None:
            # Infer type from value
            field_type = TypeInferrer.infer(value)

        if field_type is None:
            raise ValueError("Either field_type must be provided or value must be non-null for type inference.")

        # extract any custom field args
        field_kwargs = {"name": field_name, **field_args.get(field_name, {})}

        # TODO - Handle specific storage type logic?
        # if self.storage_type == StorageType.JSON:
        #     field_kwargs["as_name"] = field_name
        #     field_kwargs["name"] = f"$.{field_name.replace(' ', '')}"

        # Getting field class from type map
        field_class = self._get_field_class(field_type)

        # Creating field instance
        try:
            return field_type, field_class(**field_kwargs)
        except ValidationError as e:
            raise ValueError(f"Error creating field instance: {e}") from e

    def _ensure_unique_field_name(self, field_type: str, name: str):
        """Ensures the field name is unique within its type."""
        if any(field.name == name for field in self.fields.get(field_type, [])):
            raise ValueError(f"Field with name '{name}' already exists in {field_type} fields.")

    def add_field(self, field_type: str, **kwargs):
        """Add a field to the schema.

        Args:
            field_type: The type of field to add.
            kwargs: The keyword arguments for the field.

        Raises:
            ValueError: If the field name is not provided or already exists.
            ValueError: If there is a field validation error.
            ValueError: If an unknown field type is provided.
        """
        name = kwargs.get('name')
        if not name:
            raise ValueError(f"Field name must be provided. Received: {name}")

        # construct a new field instance from name, type, and kwargs
        _, new_field = self._create_field_instance(
            field_name=name,
            field_type=field_type,
            field_args={name: kwargs}
        )
        # final check and add to index schema
        self._ensure_unique_field_name(field_type, name)
        self.fields.setdefault(field_type, []).append(new_field)

    def generate_fields(
        self,
        data: Dict[str, Any],
        strict: bool = False,
        ignore_fields: List[str] = [],
        field_args: Dict[str, Dict[str, Any]] = {}
    ) -> Dict[str, List[Dict[str, Any]]]:
        """_summary_

        Args:
            data (Dict[str, Any]): _description_
            strict (bool, optional): _description_. Defaults to False.
            ignore_fields (List[str], optional): _description_. Defaults to [].
            field_args (Dict[str, Dict[str, Any]], optional): _description_. Defaults to {}.

        Returns:
            Dict[str, List[Dict[str, Any]]]: _description_
        """
        fields = {}
        for field_name, value in data.items():
            if self._should_ignore_field(field_name, ignore_fields):
                # ignore if specified
                continue
            try:
                field_type, new_field = self._create_field_instance(
                    field_name=field_name,
                    value=value,
                    field_args=field_args
                )
                fields.setdefault(field_type, []).append(new_field.dict(exclude_unset=True))
            except ValueError as e:
                if strict:
                    raise
                else:
                    print(f"Error inferring field type for {field_name}: {e}")
        return fields

    def remove_field(self, field_type: str, field_name: str):
        """Remove a field from the schema.

        Args:
            field_type (str): The type of field to add.
            field_name (str): The name of the field to add.
        """
        if field_type in self.fields:
            self.fields[field_type] = [
                field for field in self.fields[field_type] if field.name != field_name]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexSchema":
        """Generate an index schema object from a dictionary representation

        Args:
            data (Dict[str, Any]): Data to use building the index schema.

        Returns:
            A Schema instance.
        """
        schema = cls(**data['index'])
        for field_type, field_list in data['fields'].items():
            for field_data in field_list:
                # make use of our add field method!
                schema.add_field(field_type, **field_data)
        return schema

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

    def _check_yaml_path(self, file_path: str) -> Path:
        if not file_path.endswith(".yaml"):
            raise ValueError("Must provide a valid YAML file path")

        return Path(file_path).resolve()

    @classmethod
    def from_yaml(cls, file_path: str) -> "IndexSchema":
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
        # Check file path
        fp = cls._check_yaml_path(file_path)
        if not fp.exists():
            raise FileNotFoundError(f"Schema file {file_path} does not exist")

        with open(fp, "r") as f:
            yaml_data = yaml.safe_load(f)

        return cls.from_dict(yaml_data)

    def to_yaml(self, file_path: str, overwrite: bool = True) -> None:
        """
        Write the schema to a yaml file.

        Args:
            file_path (str): The yaml file path where the RedisVL schema is written.

        """
        # Check filepath
        fp = self._check_yaml_path(file_path)
        if fp.exists() and overwrite == False:
            raise FileExistsError(f"Schema file {file_path} already exists.")

        schema = self.to_dict()
        with open(file_path, "w+") as f:
            f.write(yaml.dump(schema, sort_keys=False))

    def _should_ignore_field(self, field_name: str, ignore_fields: List[str]) -> bool:
        """Ignore a specified field?"""
        return field_name in ignore_fields







import re
from typing import Any

class TypeInferrer:
    """
    Infers the type of a field based on its value.
    """

    GEO_PATTERN = re.compile(
        r"^\s*[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)\s*$"
    )

    TYPE_METHOD_MAP = {
        "numeric": "_is_numeric",
        "geo": "_is_geographic",
        "tag": "_is_tag",
        "text": "_is_text",
    }

    @classmethod
    def infer(cls, value: Any) -> str:
        """
        Infers the field type for a given value.

        Args:
            value: The value to infer the type of.

        Returns:
            The inferred field type as a string.

        Raises:
            ValueError: If the type cannot be inferred.
        """
        for type_name, method_name in cls.TYPE_METHOD_MAP.items():
            if getattr(cls, method_name)(value):
                return type_name
        raise ValueError(f"Unable to infer type for value: {value}")

    @classmethod
    def _is_numeric(cls, value: Any) -> bool:
        """Check if the value is numeric."""
        if not isinstance(value, (int, float, str)):
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @classmethod
    def _is_tag(cls, value: Any) -> bool:
        """Check if the value is a tag."""
        return isinstance(value, (list, set, tuple)) and all(isinstance(v, str) for v in value)

    @classmethod
    def _is_text(cls, value: Any) -> bool:
        """Check if the value is text."""
        return isinstance(value, str)

    @classmethod
    def _is_geographic(cls, value: Any) -> bool:
        """Check if the value is a geographic coordinate."""
        return isinstance(value, str) and cls.GEO_PATTERN.match(value) is not None
