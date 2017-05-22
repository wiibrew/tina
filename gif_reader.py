#this file is for more convicient gif load when doing deep learning training ot testing

import os
import re
import numpy as np
from PIL import Image


class gif_reader(object):
	"""docstring for gif_reader"""
	def __init__(self, arg):
		super(gif_reader, self).__init__()
		self.arg = arg
		self.path=arg[0]
		self.gif_array=self.get_array()
		self.N=self.gif_array.shape[0]


	def analyseImage(self):
		'''
		Pre-process pass over the image to determine the mode (full or additive).
		Necessary as assessing single frames isn't reliable. Need to know the mode 
		before processing all frames.
		'''
		path=self.path
		# print path
		im = Image.open(path)
		results = {
			'size': im.size,
			'mode': 'full',
		}
		try:
			while True:
				if im.tile:
					tile = im.tile[0]
					update_region = tile[1]
					update_region_dimensions = update_region[2:]
					if update_region_dimensions != im.size:
						results['mode'] = 'partial'
						break
				im.seek(im.tell() + 1)
		except EOFError:
			pass
		return results


	def processImage(path):
		'''
		Iterate the GIF, extracting each frame.
		'''
		mode = analyseImage(path)['mode']
		
		im = Image.open(path)

		i = 0
		p = im.getpalette()
		last_frame = im.convert('RGBA')
		
		try:
			while True:
				print "saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile)
				
				'''
				If the GIF uses local colour tables, each frame will have its own palette.
				If not, we need to apply the global palette to the new frame.
				'''
				if not im.getpalette():
					im.putpalette(p)
				
				new_frame = Image.new('RGBA', im.size)
				
				'''
				Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
				If so, we need to construct the new frame by pasting it on top of the preceding frames.
				'''
				if mode == 'partial':
					new_frame.paste(last_frame)
				
				new_frame.paste(im, (0,0), im.convert('RGBA'))
				new_frame.save('%s-%d.png' % (''.join(os.path.basename(path).split('.')[:-1]), i), 'PNG')

				i += 1
				last_frame = new_frame
				im.seek(im.tell() + 1)
		except EOFError:
			pass
	def get_array(self):
	#read gif from path, read as a n size array for deep learning training
		gif_path=self.path

		gif_array=[]

		mode = self.analyseImage()['mode']
		
		im = Image.open(gif_path)

		i = 0
		p = im.getpalette()
		last_frame = im.convert('RGBA')
		
		try:
			while True:
				# print "saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile)
				
				'''
				If the GIF uses local colour tables, each frame will have its own palette.
				If not, we need to apply the global palette to the new frame.
				'''
				if not im.getpalette():
					im.putpalette(p)
				
				new_frame = Image.new('RGBA', im.size)
				
				'''
				Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
				If so, we need to construct the new frame by pasting it on top of the preceding frames.
				'''
				if mode == 'partial':
					new_frame.paste(last_frame)
				#gif data is read here
				new_frame.paste(im, (0,0), im.convert('RGBA'))
				# new_frame.save('%s-%d.png' % (''.join(os.path.basename(path).split('.')[:-1]), i), 'PNG')
				gif_array+=[np.array(new_frame)]
				i += 1
				last_frame = new_frame
				im.seek(im.tell() + 1)
		except EOFError:
			pass

		return np.array(gif_array) 

	def gif_show(self):
	#show the gif to see the content
		return

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



			
