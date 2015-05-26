__author__ = 'GregaP'

from setuptools import setup

setup(name="NomogramGeneral",
      packages=["nomogramgeneral"],
      # Declare orangedemo package to contain widgets for the "Demo" category
      entry_points={"orange.widgets": ("Classify = nomogramgeneral")},
      )
