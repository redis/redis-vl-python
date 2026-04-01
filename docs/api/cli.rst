**********************
Command Line Interface
**********************

RedisVL provides a command line interface (CLI) called ``rvl`` for managing vector search indices. The CLI enables you to create, inspect, and delete indices directly from your terminal without writing Python code.

Installation
============

The ``rvl`` command is included when you install RedisVL.

.. code-block:: bash

    pip install redisvl

Verify the installation by running:

.. code-block:: bash

    rvl version

Connection Configuration
========================

The CLI connects to Redis using the following resolution order:

1. The ``REDIS_URL`` environment variable, if set
2. Explicit connection flags (``--host``, ``--port``, ``--url``)
3. Default values (``localhost:6379``)

**Connection Flags**

All commands that interact with Redis accept these optional flags:

.. list-table::
   :widths: 20 15 50 15
   :header-rows: 1

   * - Flag
     - Type
     - Description
     - Default
   * - ``-u``, ``--url``
     - string
     - Full Redis URL (e.g., ``redis://localhost:6379``)
     - None
   * - ``--host``
     - string
     - Redis server hostname
     - ``localhost``
   * - ``-p``, ``--port``
     - integer
     - Redis server port
     - ``6379``
   * - ``--user``
     - string
     - Redis username for authentication
     - ``default``
   * - ``-a``, ``--password``
     - string
     - Redis password for authentication
     - Empty
   * - ``--ssl``
     - flag
     - Enable SSL/TLS encryption
     - Disabled

**Examples**

Connect using environment variable:

.. code-block:: bash

    export REDIS_URL="redis://localhost:6379"
    rvl index listall

Connect with explicit host and port:

.. code-block:: bash

    rvl index listall --host myredis.example.com --port 6380

Connect with authentication and SSL:

.. code-block:: bash

    rvl index listall --user admin --password secret --ssl

Getting Help
============

All commands support the ``-h`` and ``--help`` flags to display usage information.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Flag
     - Description
   * - ``-h``, ``--help``
     - Display usage information for the command

**Examples**

.. code-block:: bash

    # Display top-level help
    rvl --help

    # Display help for a command group
    rvl index --help

    # Display help for a specific subcommand
    rvl index create --help

Running ``rvl`` without any arguments also displays the top-level help message.

.. tip::

   For a hands-on tutorial with practical examples, see the :doc:`/user_guide/cli`.

Commands
========

rvl version
-----------

Display the installed RedisVL version.

**Syntax**

.. code-block:: bash

    rvl version [OPTIONS]

**Options**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-s``, ``--short``
     - Print only the version number without additional formatting

**Examples**

.. code-block:: bash

    # Full version output
    rvl version

    # Version number only
    rvl version --short

rvl index
---------

Manage vector search indices. This command group provides subcommands for creating, inspecting, listing, and removing indices.

**Syntax**

.. code-block:: bash

    rvl index <subcommand> [OPTIONS]

**Subcommands**

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Subcommand
     - Description
   * - ``create``
     - Create a new index from a YAML schema file
   * - ``info``
     - Display detailed information about an index
   * - ``listall``
     - List all existing indices in the Redis instance
   * - ``delete``
     - Remove an index while preserving the underlying data
   * - ``destroy``
     - Remove an index and delete all associated data

rvl index create
^^^^^^^^^^^^^^^^

Create a new vector search index from a YAML schema definition.

**Syntax**

.. code-block:: bash

    rvl index create -s <schema_file> [CONNECTION_OPTIONS]

**Required Options**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-s``, ``--schema``
     - Path to the YAML schema file defining the index structure

**Example**

.. code-block:: bash

    rvl index create -s schema.yaml

**Schema File Format**

The schema file must be valid YAML with the following structure:

.. code-block:: yaml

    version: '0.1.0'

    index:
        name: my_index
        prefix: doc
        storage_type: hash

    fields:
        - name: content
          type: text
        - name: embedding
          type: vector
          attrs:
            dims: 768
            algorithm: hnsw
            distance_metric: cosine

rvl index info
^^^^^^^^^^^^^^

Display detailed information about an existing index, including field definitions and index options.

**Syntax**

.. code-block:: bash

    rvl index info (-i <index_name> | -s <schema_file>) [OPTIONS]

**Options**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-i``, ``--index``
     - Name of the index to inspect
   * - ``-s``, ``--schema``
     - Path to the schema file (alternative to specifying index name)

**Example**

.. code-block:: bash

    rvl index info -i my_index

**Output**

The command displays two tables:

1. **Index Information** containing the index name, storage type, key prefixes, index options, and indexing status
2. **Index Fields** listing each field with its name, attribute, type, and any additional field options

rvl index listall
^^^^^^^^^^^^^^^^^

List all vector search indices in the connected Redis instance.

**Syntax**

.. code-block:: bash

    rvl index listall [CONNECTION_OPTIONS]

**Example**

.. code-block:: bash

    rvl index listall

**Output**

Returns a numbered list of all index names:

.. code-block:: text

    Indices:
    1. products_index
    2. documents_index
    3. embeddings_index

rvl index delete
^^^^^^^^^^^^^^^^

Remove an index from Redis while preserving the underlying data. Use this when you want to rebuild an index with a different schema without losing your data.

**Syntax**

.. code-block:: bash

    rvl index delete (-i <index_name> | -s <schema_file>) [CONNECTION_OPTIONS]

**Options**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-i``, ``--index``
     - Name of the index to delete
   * - ``-s``, ``--schema``
     - Path to the schema file (alternative to specifying index name)

**Example**

.. code-block:: bash

    rvl index delete -i my_index

rvl index destroy
^^^^^^^^^^^^^^^^^

Remove an index and permanently delete all associated data from Redis. This operation cannot be undone.

**Syntax**

.. code-block:: bash

    rvl index destroy (-i <index_name> | -s <schema_file>) [CONNECTION_OPTIONS]

**Options**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-i``, ``--index``
     - Name of the index to destroy
   * - ``-s``, ``--schema``
     - Path to the schema file (alternative to specifying index name)

**Example**

.. code-block:: bash

    rvl index destroy -i my_index

.. warning::

   This command permanently deletes both the index and all documents stored with the index prefix. Ensure you have backups before running this command.

rvl stats
---------

Display statistics about an existing index, including document counts, memory usage, and indexing performance metrics.

**Syntax**

.. code-block:: bash

    rvl stats (-i <index_name> | -s <schema_file>) [OPTIONS]

**Options**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-i``, ``--index``
     - Name of the index to query
   * - ``-s``, ``--schema``
     - Path to the schema file (alternative to specifying index name)

**Example**

.. code-block:: bash

    rvl stats -i my_index

**Statistics Reference**

The command returns the following metrics:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Metric
     - Description
   * - ``num_docs``
     - Total number of indexed documents
   * - ``num_terms``
     - Number of distinct terms in text fields
   * - ``max_doc_id``
     - Highest internal document ID
   * - ``num_records``
     - Total number of index records
   * - ``percent_indexed``
     - Percentage of documents fully indexed
   * - ``hash_indexing_failures``
     - Number of documents that failed to index
   * - ``number_of_uses``
     - Number of times the index has been queried
   * - ``bytes_per_record_avg``
     - Average bytes per index record
   * - ``doc_table_size_mb``
     - Document table size in megabytes
   * - ``inverted_sz_mb``
     - Inverted index size in megabytes
   * - ``key_table_size_mb``
     - Key table size in megabytes
   * - ``offset_bits_per_record_avg``
     - Average offset bits per record
   * - ``offset_vectors_sz_mb``
     - Offset vectors size in megabytes
   * - ``offsets_per_term_avg``
     - Average offsets per term
   * - ``records_per_doc_avg``
     - Average records per document
   * - ``sortable_values_size_mb``
     - Sortable values size in megabytes
   * - ``total_indexing_time``
     - Total time spent indexing in milliseconds
   * - ``total_inverted_index_blocks``
     - Number of inverted index blocks
   * - ``vector_index_sz_mb``
     - Vector index size in megabytes

Exit Codes
==========

The CLI returns the following exit codes:

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Code
     - Description
   * - ``0``
     - Command completed successfully
   * - ``1``
     - Command failed due to missing required arguments or invalid input

Related Resources
=================

- :doc:`/user_guide/cli` for a tutorial-style walkthrough
- :doc:`schema` for YAML schema format details
- :doc:`searchindex` for the Python ``SearchIndex`` API

