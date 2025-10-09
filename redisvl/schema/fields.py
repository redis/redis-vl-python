"""
RedisVL Fields, FieldAttributes, and Enums

Reference Redis search source documentation as needed: https://redis.io/commands/ft.create/
Reference Redis vector search documentation as needed: https://redis.io/docs/interact/search-and-query/advanced-concepts/vectors/
"""

from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from redis.commands.search.field import Field as RedisField
from redis.commands.search.field import GeoField as RedisGeoField
from redis.commands.search.field import NumericField as RedisNumericField
from redis.commands.search.field import TagField as RedisTagField
from redis.commands.search.field import TextField as RedisTextField
from redis.commands.search.field import VectorField as RedisVectorField

from redisvl.utils.utils import norm_cosine_distance, norm_l2_distance

VECTOR_NORM_MAP = {
    "COSINE": norm_cosine_distance,
    "L2": norm_l2_distance,
    "IP": None,  # normalized inner product is cosine similarity by definition
}


class FieldTypes(str, Enum):
    TAG = "tag"
    TEXT = "text"
    NUMERIC = "numeric"
    GEO = "geo"
    VECTOR = "vector"


class VectorDistanceMetric(str, Enum):
    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"


class VectorDataType(str, Enum):
    BFLOAT16 = "BFLOAT16"
    FLOAT16 = "FLOAT16"
    FLOAT32 = "FLOAT32"
    FLOAT64 = "FLOAT64"
    INT8 = "INT8"
    UINT8 = "UINT8"


class VectorIndexAlgorithm(str, Enum):
    FLAT = "FLAT"
    HNSW = "HNSW"
    SVS_VAMANA = "SVS-VAMANA"


class CompressionType(str, Enum):
    """Vector compression types for SVS-VAMANA algorithm"""

    LVQ4 = "LVQ4"
    LVQ4x4 = "LVQ4x4"
    LVQ4x8 = "LVQ4x8"
    LVQ8 = "LVQ8"
    LeanVec4x8 = "LeanVec4x8"
    LeanVec8x8 = "LeanVec8x8"


### Field Attributes ###


class BaseFieldAttributes(BaseModel):
    """Base field attributes shared by other lexical fields"""

    sortable: bool = Field(default=False)
    """Enable faster result sorting on the field at runtime"""
    index_missing: bool = Field(default=False)
    """Allow indexing and searching for missing values (documents without the field)"""
    no_index: bool = Field(default=False)
    """Store field without indexing it (requires sortable=True or field is ignored)"""


class TextFieldAttributes(BaseFieldAttributes):
    """Full text field attributes"""

    weight: float = Field(default=1)
    """Declares the importance of this field when calculating results"""
    no_stem: bool = Field(default=False)
    """Disable stemming on the text field during indexing"""
    withsuffixtrie: bool = Field(default=False)
    """Keep a suffix trie with all terms which match the suffix to optimize certain queries"""
    phonetic_matcher: Optional[str] = None
    """Used to perform phonetic matching during search"""
    index_empty: bool = Field(default=False)
    """Allow indexing and searching for empty strings"""
    unf: bool = Field(default=False)
    """Un-normalized form - disable normalization on sortable fields (only applies when sortable=True)"""


class TagFieldAttributes(BaseFieldAttributes):
    """Tag field attributes"""

    separator: str = Field(default=",")
    """Indicates how the text in the original attribute is split into individual tags"""
    case_sensitive: bool = Field(default=False)
    """Treat text as case sensitive or not. By default, tag characters are converted to lowercase"""
    withsuffixtrie: bool = Field(default=False)
    """Keep a suffix trie with all terms which match the suffix to optimize certain queries"""
    index_empty: bool = Field(default=False)
    """Allow indexing and searching for empty strings"""


class NumericFieldAttributes(BaseFieldAttributes):
    """Numeric field attributes"""

    unf: bool = Field(default=False)
    """Un-normalized form - disable normalization on sortable fields (only applies when sortable=True)"""


class GeoFieldAttributes(BaseFieldAttributes):
    """Numeric field attributes"""

    pass


class BaseVectorFieldAttributes(BaseModel):
    """Base vector field attributes shared by both FLAT and HNSW fields"""

    dims: int
    """Dimensionality of the vector embeddings field"""
    algorithm: VectorIndexAlgorithm
    """The indexing algorithm for the field: HNSW or FLAT"""
    datatype: VectorDataType = Field(default=VectorDataType.FLOAT32)
    """The float datatype for the vector embeddings"""
    distance_metric: VectorDistanceMetric = Field(default=VectorDistanceMetric.COSINE)
    """The distance metric used to measure query relevance"""
    initial_cap: Optional[int] = None
    """Initial vector capacity in the index affecting memory allocation size of the index"""
    index_missing: bool = Field(default=False)
    """Allow indexing and searching for missing values (documents without the field)"""

    @field_validator("algorithm", "datatype", "distance_metric", mode="before")
    @classmethod
    def uppercase_strings(cls, v):
        """Validate that provided values are cast to uppercase"""
        return v.upper()

    @property
    def field_data(self) -> Dict[str, Any]:
        """Select attributes required by the Redis API"""
        field_data = {
            "TYPE": self.datatype,
            "DIM": self.dims,
            "DISTANCE_METRIC": self.distance_metric,
        }
        if self.initial_cap is not None:  # Only include it if it's set
            field_data["INITIAL_CAP"] = self.initial_cap
        if self.index_missing:  # Only include it if it's set
            field_data["INDEXMISSING"] = True
        return field_data


class FlatVectorFieldAttributes(BaseVectorFieldAttributes):
    """FLAT vector field attributes"""

    algorithm: Literal[VectorIndexAlgorithm.FLAT] = VectorIndexAlgorithm.FLAT
    """The indexing algorithm for the vector field"""
    block_size: Optional[int] = None
    """Block size to hold amount of vectors in a contiguous array. This is useful when the index is dynamic with respect to addition and deletion"""


class HNSWVectorFieldAttributes(BaseVectorFieldAttributes):
    """HNSW vector field attributes"""

    algorithm: Literal[VectorIndexAlgorithm.HNSW] = VectorIndexAlgorithm.HNSW
    """The indexing algorithm for the vector field"""
    m: int = Field(default=16)
    """Number of max outgoing edges for each graph node in each layer"""
    ef_construction: int = Field(default=200)
    """Number of max allowed potential outgoing edges candidates for each node in the graph during build time"""
    ef_runtime: int = Field(default=10)
    """Number of maximum top candidates to hold during KNN search"""
    epsilon: float = Field(default=0.01)
    """Relative factor that sets the boundaries in which a range query may search for candidates"""


class SVSVectorFieldAttributes(BaseVectorFieldAttributes):
    """SVS-VAMANA vector field attributes with optional compression support"""

    algorithm: Literal[VectorIndexAlgorithm.SVS_VAMANA] = (
        VectorIndexAlgorithm.SVS_VAMANA
    )
    """The indexing algorithm for the vector field"""

    # SVS-VAMANA graph parameters
    graph_max_degree: int = Field(default=40)
    """Maximum degree of the Vamana graph (number of edges per node)"""
    construction_window_size: int = Field(default=250)
    """Size of the candidate list during graph construction"""
    search_window_size: int = Field(default=20)
    """Size of the candidate list during search"""
    epsilon: float = Field(default=0.01)
    """Relative factor for range query boundaries"""

    # SVS-VAMANA compression parameters (optional, to be implemented)
    compression: Optional[CompressionType] = None
    """Vector compression type (LVQ or LeanVec)"""
    reduce: Optional[int] = None
    """Reduced dimensionality for LeanVec compression (must be < dims)"""
    training_threshold: Optional[int] = None
    """Minimum number of vectors required before compression training"""

    @model_validator(mode="after")
    def validate_svs_params(self):
        """Validate SVS-VAMANA specific constraints"""
        # Datatype validation: SVS only supports FLOAT16 and FLOAT32
        if self.datatype not in (VectorDataType.FLOAT16, VectorDataType.FLOAT32):
            raise ValueError(
                f"SVS-VAMANA only supports FLOAT16 and FLOAT32 datatypes. "
                f"Got: {self.datatype}. "
                f"Unsupported types: BFLOAT16, FLOAT64, INT8, UINT8."
            )

        # Reduce validation: must be less than dims and only valid with LeanVec
        if self.reduce is not None:
            if self.reduce >= self.dims:
                raise ValueError(
                    f"reduce ({self.reduce}) must be less than dims ({self.dims})"
                )

            # Validate that reduce is only used with LeanVec compression
            if self.compression is None:
                raise ValueError(
                    "reduce parameter requires compression to be set. "
                    "Use LeanVec4x8 or LeanVec8x8 compression with reduce."
                )

            if not self.compression.value.startswith("LeanVec"):
                raise ValueError(
                    f"reduce parameter is only supported with LeanVec compression types. "
                    f"Got compression={self.compression.value}. "
                    f"Either use LeanVec4x8/LeanVec8x8 or remove the reduce parameter."
                )

        # Phase C: Add warning for LeanVec without reduce
        # if self.compression and self.compression.value.startswith("LeanVec") and not self.reduce:
        #     logger.warning(
        #         f"LeanVec compression selected without 'reduce'. "
        #         f"Consider setting reduce={self.dims//2} for better performance"
        #     )

        return self


### Field Classes ###


class BaseField(BaseModel):
    """Base field"""

    name: str
    """Field name"""
    type: str
    """Field type"""
    path: Optional[str] = None
    """Field path (within JSON object)"""
    attrs: Optional[Union[BaseFieldAttributes, BaseVectorFieldAttributes]] = None
    """Specified field attributes"""

    def _handle_names(self) -> Tuple[str, Optional[str]]:
        """Helper to handle field naming with path support"""
        if self.path:
            return self.path, self.name
        return self.name, None

    def as_redis_field(self) -> RedisField:
        """Convert schema field to Redis Field object"""
        raise NotImplementedError("Must be implemented by field subclasses")


class TextField(BaseField):
    """Text field supporting a full text search index"""

    type: Literal[FieldTypes.TEXT] = FieldTypes.TEXT
    attrs: TextFieldAttributes = Field(default_factory=TextFieldAttributes)

    def as_redis_field(self) -> RedisField:
        name, as_name = self._handle_names()
        # Build arguments for RedisTextField
        kwargs: Dict[str, Any] = {
            "weight": self.attrs.weight,  # type: ignore
            "no_stem": self.attrs.no_stem,  # type: ignore
            "sortable": self.attrs.sortable,
        }

        # Only add as_name if it's not None
        if as_name is not None:
            kwargs["as_name"] = as_name

        # Only add phonetic_matcher if it's not None
        if self.attrs.phonetic_matcher is not None:  # type: ignore
            kwargs["phonetic_matcher"] = self.attrs.phonetic_matcher  # type: ignore

        # Add INDEXMISSING if enabled
        if self.attrs.index_missing:  # type: ignore
            kwargs["index_missing"] = True

        # Add INDEXEMPTY if enabled
        if self.attrs.index_empty:  # type: ignore
            kwargs["index_empty"] = True

        # Add NOINDEX if enabled
        if self.attrs.no_index:  # type: ignore
            kwargs["no_index"] = True

        field = RedisTextField(name, **kwargs)

        # Add UNF support (only when sortable)
        # UNF must come before NOINDEX in the args_suffix
        if self.attrs.unf and self.attrs.sortable:  # type: ignore
            if "NOINDEX" in field.args_suffix:
                # Insert UNF before NOINDEX
                noindex_idx = field.args_suffix.index("NOINDEX")
                field.args_suffix.insert(noindex_idx, "UNF")
            else:
                # No NOINDEX, append normally
                field.args_suffix.append("UNF")

        return field


class TagField(BaseField):
    """Tag field for simple boolean-style filtering"""

    type: Literal[FieldTypes.TAG] = FieldTypes.TAG
    attrs: TagFieldAttributes = Field(default_factory=TagFieldAttributes)

    def as_redis_field(self) -> RedisField:
        name, as_name = self._handle_names()
        # Build arguments for RedisTagField
        kwargs: Dict[str, Any] = {
            "separator": self.attrs.separator,  # type: ignore
            "case_sensitive": self.attrs.case_sensitive,  # type: ignore
            "sortable": self.attrs.sortable,
        }

        # Only add as_name if it's not None
        if as_name is not None:
            kwargs["as_name"] = as_name

        # Add INDEXMISSING if enabled
        if self.attrs.index_missing:  # type: ignore
            kwargs["index_missing"] = True

        # Add INDEXEMPTY if enabled
        if self.attrs.index_empty:  # type: ignore
            kwargs["index_empty"] = True

        # Add NOINDEX if enabled
        if self.attrs.no_index:  # type: ignore
            kwargs["no_index"] = True

        return RedisTagField(name, **kwargs)


class NumericField(BaseField):
    """Numeric field for numeric range filtering"""

    type: Literal[FieldTypes.NUMERIC] = FieldTypes.NUMERIC
    attrs: NumericFieldAttributes = Field(default_factory=NumericFieldAttributes)

    def as_redis_field(self) -> RedisField:
        name, as_name = self._handle_names()
        # Build arguments for RedisNumericField
        kwargs: Dict[str, Any] = {
            "sortable": self.attrs.sortable,
        }

        # Only add as_name if it's not None
        if as_name is not None:
            kwargs["as_name"] = as_name

        # Add INDEXMISSING if enabled
        if self.attrs.index_missing:  # type: ignore
            kwargs["index_missing"] = True

        # Add NOINDEX if enabled
        if self.attrs.no_index:  # type: ignore
            kwargs["no_index"] = True

        field = RedisNumericField(name, **kwargs)

        # Add UNF support (only when sortable)
        # UNF must come before NOINDEX in the args_suffix
        if self.attrs.unf and self.attrs.sortable:  # type: ignore
            if "NOINDEX" in field.args_suffix:
                # Insert UNF before NOINDEX
                noindex_idx = field.args_suffix.index("NOINDEX")
                field.args_suffix.insert(noindex_idx, "UNF")
            else:
                # No NOINDEX, append normally
                field.args_suffix.append("UNF")

        return field


class GeoField(BaseField):
    """Geo field with a geo-spatial index for location based search"""

    type: Literal[FieldTypes.GEO] = FieldTypes.GEO
    attrs: GeoFieldAttributes = Field(default_factory=GeoFieldAttributes)

    def as_redis_field(self) -> RedisField:
        name, as_name = self._handle_names()
        # Build arguments for RedisGeoField
        kwargs: Dict[str, Any] = {
            "sortable": self.attrs.sortable,
        }

        # Only add as_name if it's not None
        if as_name is not None:
            kwargs["as_name"] = as_name

        # Add INDEXMISSING if enabled
        if self.attrs.index_missing:  # type: ignore
            kwargs["index_missing"] = True

        # Add NOINDEX if enabled
        if self.attrs.no_index:  # type: ignore
            kwargs["no_index"] = True

        return RedisGeoField(name, **kwargs)


class FlatVectorField(BaseField):
    """Vector field with a FLAT index (brute force nearest neighbors search)"""

    type: Literal[FieldTypes.VECTOR] = FieldTypes.VECTOR
    attrs: FlatVectorFieldAttributes

    def as_redis_field(self) -> RedisField:
        # grab base field params and augment with flat-specific fields
        name, as_name = self._handle_names()
        field_data = self.attrs.field_data
        if self.attrs.block_size is not None:
            field_data["BLOCK_SIZE"] = self.attrs.block_size
        return RedisVectorField(name, self.attrs.algorithm, field_data, as_name=as_name)


class HNSWVectorField(BaseField):
    """Vector field with an HNSW index (approximate nearest neighbors search)"""

    type: Literal["vector"] = "vector"
    attrs: HNSWVectorFieldAttributes

    def as_redis_field(self) -> RedisField:
        # grab base field params and augment with hnsw-specific fields
        name, as_name = self._handle_names()
        field_data = self.attrs.field_data
        field_data.update(
            {
                "M": self.attrs.m,
                "EF_CONSTRUCTION": self.attrs.ef_construction,
                "EF_RUNTIME": self.attrs.ef_runtime,
                "EPSILON": self.attrs.epsilon,
            }
        )
        return RedisVectorField(name, self.attrs.algorithm, field_data, as_name=as_name)


class SVSVectorField(BaseField):
    """Vector field with an SVS-VAMANA index"""

    type: Literal[FieldTypes.VECTOR] = FieldTypes.VECTOR
    attrs: SVSVectorFieldAttributes

    def as_redis_field(self) -> RedisField:
        name, as_name = self._handle_names()
        field_data = self.attrs.field_data
        field_data.update(
            {
                "GRAPH_MAX_DEGREE": self.attrs.graph_max_degree,
                "CONSTRUCTION_WINDOW_SIZE": self.attrs.construction_window_size,
                "SEARCH_WINDOW_SIZE": self.attrs.search_window_size,
                "EPSILON": self.attrs.epsilon,
            }
        )
        # Add compression parameters if specified
        if self.attrs.compression is not None:
            field_data["COMPRESSION"] = self.attrs.compression
        if self.attrs.reduce is not None:
            field_data["REDUCE"] = self.attrs.reduce
        if self.attrs.training_threshold is not None:
            field_data["TRAINING_THRESHOLD"] = self.attrs.training_threshold
        return RedisVectorField(name, self.attrs.algorithm, field_data, as_name=as_name)


FIELD_TYPE_MAP = {
    "tag": TagField,
    "text": TextField,
    "numeric": NumericField,
    "geo": GeoField,
}

VECTOR_FIELD_TYPE_MAP = {
    "flat": FlatVectorField,
    "hnsw": HNSWVectorField,
    "svs-vamana": SVSVectorField,
}


class FieldFactory:
    """Factory class to create fields from client data and kwargs."""

    @classmethod
    def pick_vector_field_type(cls, attrs: Dict[str, Any]) -> Type[BaseField]:
        """Get the vector field type from the field data."""
        if "algorithm" not in attrs:
            raise ValueError("Must provide algorithm param for the vector field.")

        if "dims" not in attrs:
            raise ValueError("Must provide dims param for the vector field.")

        algorithm = attrs["algorithm"].lower()
        if algorithm not in VECTOR_FIELD_TYPE_MAP:
            raise ValueError(f"Unknown vector field algorithm: {algorithm}")

        return VECTOR_FIELD_TYPE_MAP[algorithm]  # type: ignore

    @classmethod
    def create_field(
        cls,
        type: str,
        name: str,
        attrs: Dict[str, Any] = {},
        path: Optional[str] = None,
    ) -> BaseField:
        """Create a field of a given type with provided attributes."""

        if type == "vector":
            field_class = cls.pick_vector_field_type(attrs)
        else:
            if type not in FIELD_TYPE_MAP:
                raise ValueError(f"Unknown field type: {type}")
            field_class = FIELD_TYPE_MAP[type]  # type: ignore

        return field_class.model_validate(
            {
                "name": name,
                "path": path,
                "attrs": attrs,
            }
        )
