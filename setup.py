#!/usr/bin/env python
import os
from setuptools import setup, find_packages


root_dir = os.path.abspath(os.path.dirname(__file__))

infos = {
         'name': 'bruker2nifti',
         'version': '0.0.0',
         'description': 'From raw Bruker to nifti, home-made converter.',
         'web_infos' : '',
         'repository': {
                        'type': 'git',
                        'url': ''
                       },
         'author': 'sebastiano ferraris',
         'author_email': 'sebastiano.ferraris@gmail.com',
         'dependencies': {
                          # requirements.txt file automatically generated using pipreqs.
                          'python' : '{0}/requirements.txt'.format(root_dir)
                          }
         }

setup(name=infos['name'],
      version=infos['version'],
      description=infos['description'],
      author=infos['author'],
      author_email=infos['author_email'],
      url=infos['repository']['url'],
      packages=find_packages()
      )
