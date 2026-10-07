# FSDBAPI third-party runtime dependencies

This directory contains the minimum Verdi FsdbReader files required by the
PowerModeling FSDBAPI tool.

Bundled files:

```text
verdi/FsdbReader/ffrAPI.h
verdi/FsdbReader/ffrKit.h
verdi/FsdbReader/fsdbShr.h
verdi/FsdbReader/FILE_LIST
verdi/FsdbReader/linux64/COMPILER_DEF
verdi/FsdbReader/linux64/libnffr.so
verdi/FsdbReader/linux64/libnsys.so
```

The FSDBAPI Makefile prefers this bundled SDK path by default:

```text
third_party/verdi/FsdbReader
```

If a target platform needs to use a site-installed Verdi instead, override:

```bash
make FSDB_READER_DIR=/path/to/verdi/share/FsdbReader
```

or:

```bash
make VERDI_HOME=/path/to/verdi/R-2020.12-SP2
```

Check the linked runtime libraries with:

```bash
ldd build/fsdb_to_tog_cache
```

Note: Verdi/FsdbReader files are proprietary EDA components. Make sure delivery
and redistribution comply with the applicable tool license.
