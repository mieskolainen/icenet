::

  ██  ██████ ███████ ███    ██ ███████ ████████
  ██ ██      ██      ████   ██ ██         ██   
  ██ ██      █████   ██ ██  ██ █████      ██   
  ██ ██      ██      ██  ██ ██ ██         ██   
  ██  ██████ ███████ ██   ████ ███████    ██   

A deep learning driven library for high energy physics and beyond.

https://github.com/mieskolainen/icenet


First steps
===================

Start with the installation:

``docs/source/notes/installation.rst``


For end-to-end deep learning examples, see e.g. github actions (CI) workflows under

``.github/workflows``


References
===================

If you use this work in your research -- especially if you find algorithms, their application or ideas novel,
please include a citation:

::

    @software{icenet,
      author  = "{Mikael Mieskolainen}",
      title   = "ICENET: a deep learning library for HEP",
      url     = "https://github.com/mieskolainen/icenet",
      version = {X.Y.Z},
      date    = {2024-05-30},
    }


Contents
===================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/introduction
   notes/markup

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/icebrk
   modules/icedqcd
   modules/icefit
   modules/icehgcal
   modules/icehnl
   modules/iceid
   modules/icemc
   modules/icenet
   modules/iceplot
   modules/icetrg


Indices and Tables
===================

* :ref:`genindex`
* :ref:`modindex`
