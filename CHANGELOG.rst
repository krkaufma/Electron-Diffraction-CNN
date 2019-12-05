=========
Changelog
=========

Version 0.1.0
=============
- Initial project structure established.

Version 0.1.1
=============
- Added a model "plugin" system. This allows parallel model development with reasonable version control.


Version 0.1.2
=============
- Added script to create manifest file. Manifest file is used to manage all of the data for the project -- it's a central place to look up the location of each image file and to understand it's known metadata.

- [ ] TODO: need to refactor `make_data.py`, i.e. data reading module to use the manifest file instead of rely on a specific file structure.

Version 0.1.3
=============
- Added notebooks module for interactive investigations. First investigation: Model interpretability.
- Fixed excessive warnings when reading in TIFF images.
