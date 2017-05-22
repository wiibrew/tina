#this file is for more convicient gif load when doing deep learning training ot testing

import os
import re
import numpy as np

class gif_reader(object):
	"""docstring for gif_reader"""
	def __init__(self, arg):
		super(gif_reader, self).__init__()
		self.arg = arg
		self.path=arg[1]
		self.N=int(arg[2])
	def get_array(self):
	#read gif from path, read as a n size array for deep learning training
		gif_path=self.path

		return gif_array

	def gif_show(self):
	#show the gif to see the content

	def get_tags(self):
	#obtain gif tags for futhre processing
		tag_list=[]

		return tag_list

	def get_frames_num(self):
	#get the number of frames of the gif
		filename=self.path
	    frames = 0
	    with open(filename, 'rb') as f:
	        if f.read(6) not in ('GIF87a', 'GIF89a'):
	            raise GIFError('not a valid GIF file')
	        f.seek(4, 1)
	        def skip_color_table(flags):
	            if flags & 0x80: f.seek(3 << ((flags & 7) + 1), 1)
	        flags = ord(f.read(1))
	        f.seek(2, 1)
	        skip_color_table(flags)
	        while True:
	            block = f.read(1)
	            if block == ';': break
	            if block == '!': f.seek(1, 1)
	            elif block == ',':
	                frames += 1
	                f.seek(8, 1)
	                skip_color_table(ord(f.read(1)))
	                f.seek(1, 1)
	            else: raise GIFError('unknown block type')
	            while True:
	                l = ord(f.read(1))
	                if not l: break
	                f.seek(l, 1)
	    return frames



			
