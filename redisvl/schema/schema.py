import yaml

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
from redisvl.utils.utils import (
    convert_bytes,
    make_dict,
)


class StorageType(Enum):
    HASH = "hash"
    JSON = "json"


class IndexModel(BaseModel):
    """Represents the schema for an index, including its name, optional prefix,
    and the storage type used."""
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

    def add(self, field: Union[BaseField, BaseVectorField]):
        if isinstance(field, TagField):
            self.tag.append(field)
        elif isinstance(field, TextField):
            self.text.append(field)
        elif isinstance(field, NumericField):
            self.numeric.append(field)
        elif isinstance(field, GeoField):
            self.geo.append(field)
        elif isinstance(field, BaseVectorField):
            self.vector.append(field)
        else:
            raise TypeError(f"Must provide a valid field type, received {field}")
        return self


class Schema:

    def __init__(
        self,
        index: Union[Dict[str, Any], IndexModel],
        fields: Union[Dict[str, List[Any]], FieldsModel]
    ):
        self._index = self._validate_index_model(index)
        self._fields = self._validate_fields_model(fields)

    def _validate_index_model(self, index: Union[Dict[str, Any], IndexModel]) -> IndexModel:
        """
        Validate the index model schema.
        """
        try:
            if isinstance(index, dict):
                return IndexModel(**index)
            elif isinstance(index, IndexModel):
                return index
            else:
                raise TypeError("Index must be an IndexModel instance or a dictionary.")
        except ValidationError as e:
            raise ValueError(f"Invalid index model: {e}.") from e
        except Exception as e:
            raise ValueError(f"Failed to create index model: {e}.") from e

    def _validate_fields_model(self, fields: Union[Dict[str, Any], FieldsModel]) -> FieldsModel:
        """
        Validate the fields model schema.
        """
        try:
            if isinstance(fields, dict):
                return FieldsModel(**fields)
            elif isinstance(fields, FieldsModel):
                return fields
            else:
                raise TypeError("Fields must be a FieldsModel instance or a dictionary.")
        except ValidationError as e:
            raise ValueError(f"Invalid fields model: {e}") from e
        except Exception as e:
            raise ValueError("Failed to create fields model.") from e

    @property
    def index_name(self) -> str:
        return self._index.name

    @property
    def index_prefix(self) -> str:
        return self._index.prefix

    @property
    def key_separator(self) -> str:
        return self._index.key_separator

    @property
    def storage_type(self) -> str:
        return self._index.storage_type

    @property
    def index_fields(self) -> list:
        redis_fields = []
        for field_name in self._fields.__fields__.keys():
            field_group = getattr(self._fields, field_name)
            if field_group is not None:
                for field in field_group:
                    redis_fields.append(field.as_field())
        return redis_fields

    @classmethod
    def from_params(
        cls,
        name: str,
        prefix: str = "rvl",
        key_separator: str = ":",
        storage_type: str = "hash",
        fields: Union[FieldsModel, Dict[str, List[Any]]] = {},
        **kwargs
    ):
        """
        Create a Schema instance from provided parameters.
        Args:
            name: The index name.
            prefix: The index prefix.
            key_separator: The key separator.
            storage_type: The storage type.
            fields: The field definitions.
        Returns:
            A Schema instance.
        """
        index = {
            "name": name,
            "prefix": prefix,
            "key_separator": key_separator,
            "storage_type": storage_type
        }
        return cls(index=index, fields=fields, **kwargs)

    @classmethod
    def from_db(cls, client, name: str, **kwargs):
        # TODO - eventually load a full object from here???
        # NOTE - eventually everything returned by redis ft.info output
        # client = get_redis_connection(redis_url, **connection_args)
        # TODO - does not handle sync/async yet...
        info = convert_bytes(client.ft(name).info())
        index_definition = make_dict(info["index_definition"])
        storage_type = index_definition["key_type"].lower()
        prefix = index_definition["prefixes"][0]
        # TODO - we cant even get key_separator or fields here by default... maybe use kwargs?
        key_separator = kwargs.pop("key_separator", ":")
        fields = kwargs.pop("fields", {})
        return cls.from_params(**{
            "name": name,
            "prefix": prefix,
            "storage_type": storage_type,
            "key_separator": key_separator,
            "fields": fields,
            **kwargs
        })

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

    @classmethod
    def from_data(
        cls,
        data: Dict[str, Any],
        strict: bool = False,
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a RedisVL schema from data.

        Args:
            data (Dict[str, Any]): Metadata object to validate and
                generate schema.
            strict (bool, optional): Whether to generate schema in strict
                mode. Defaults to False.

        Raises:
            ValueError: Unable to determine schema field type for a
                key-value pair.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Output metadata schema.
        """
        schema_fields = FieldsModel()

        for key, value in data.items():
            field_class = self._infer_type(value)

            if not field_class or not isinstance(field_class, (BaseField, BaseVectorField)):
                if strict:
                    raise ValueError(
                        f"Unable to determine field type for key '{key}' with"
                        f" value '{value}'"
                    )
                print(
                    f"Warning: Unable to determine field type for key '{key}'"
                    f" with value '{value}'"
                )
                continue

            # add the field to the schema fields object
            # TODO how to specify other params???
            schema_fields.add(field_class(name=key))

        return cls.from_params(fields=schema_fields, **kwargs)


    def dump(self) -> Dict[str, Any]:
        """
        Dump the RedisVL schema to a dictionary.

        Returns:
            The RedisVL schema as a dictionary.
        """
        return {
            "index": self._index.dict(),
            "fields": self._fields.dict()
        }

    def write_to_yaml(self, path: str) -> None:
        """
        Write the schema to a yaml file.

        Args:
            path (str): The yaml file path where the schema will be written.

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

