# pathlib.Path.unlink() to remove a file or directory
import sys
from PIL import Image 
import pathlib

def merge_image_colab():
  ##### first graph #####
  images = [Image.open(x) for x in ["5-1.png","5-2.png"]]
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test1.png')

  images = [Image.open(x) for x in ["5-3.png","5-4.png"]]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test2.png')

  images = [Image.open(x) for x in ['test1.png','test2.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = max(widths)
  max_height = sum(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (0,x_offset))
    x_offset += im.size[1]
  

  new_im.save('results/Epidemic impacts.png')

  ##### second graph #####

  images = [Image.open(x) for x in ["1.png","2.png"]]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test3.png')

  images = [Image.open(x) for x in ["3-1.png","3-2.png"]]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test4.png')

  images = [Image.open(x) for x in ['test3.png','test4.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = max(widths)
  max_height = sum(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (0,x_offset))
    x_offset += im.size[1]

  new_im.save('results/Economic impacts.png')

  # delete images
  for i in ["5-1.png","5-2.png",\
            "5-3.png","5-4.png",\
            "1.png","2.png",\
            "3-1.png","3-2.png",\
            'test1.png','test2.png', 'test3.png', 'test4.png']:
    f = pathlib.Path(i)
    f.unlink()

def merge_image_scenario(table):
  ##### first graph #####
  images = [Image.open(x) for x in ["table-%s-5-1.png"%table,"table-%s-5-2.png"%table]]
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test1.png')

  images = [Image.open(x) for x in ["table-%s-5-3.png"%table,"table-%s-5-4.png"%table]]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test2.png')

  images = [Image.open(x) for x in ['test1.png','test2.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = max(widths)
  max_height = sum(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (0,x_offset))
    x_offset += im.size[1]
  

  new_im.save('Results.png')

  ##### second graph #####

  images = [Image.open(x) for x in ["table-%s-1.png"%table,"table-%s-2.png"%table]]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test3.png')

  images = [Image.open(x) for x in ["table-%s-3-1.png"%table,"table-%s-3-2.png"%table]]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test4.png')

  images = [Image.open(x) for x in ['test3.png','test4.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = max(widths)
  max_height = sum(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (0,x_offset))
    x_offset += im.size[1]

  new_im.save('Decision.png')

  # delete images
  for i in ["table-%s-5-1.png"%table,"table-%s-5-2.png"%table,\
            "table-%s-5-3.png"%table,"table-%s-5-4.png"%table,\
            "table-%s-1.png"%table,"table-%s-2.png"%table,\
            "table-%s-3-1.png"%table,"table-%s-3-2.png"%table,\
            'test1.png','test2.png', 'test3.png', 'test4.png']:
    f = pathlib.Path(i)
    f.unlink()
    

def merge_image():
  ##combine 2, 3, 4
  #img1 = Image.open('1.png')
  images = [Image.open(x) for x in ['4.png','1.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test1.png')

  images = [Image.open(x) for x in ['2.png','3.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test2.png')

  images = [Image.open(x) for x in ['test1.png','test2.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = max(widths)
  max_height = sum(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (0,x_offset))
    x_offset += im.size[1]

  new_im.save('Decisions.png')

  

  images = [Image.open(x) for x in ['5.png','6.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test3.png')

  images = [Image.open(x) for x in ['7.png','8.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test4.png')

  images = [Image.open(x) for x in ['test3.png','test4.png']]#
  widths, heights = zip(*(i.size for i in images))

  total_width = max(widths)
  max_height = sum(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (0,x_offset))
    x_offset += im.size[1]

  new_im.save('Results.png')

# delete images
  for i in ['test1.png','test2.png', 'test3.png','test4.png',\
            '1.png', '2.png','3.png', '4.png', '5.png', '6.png',\
            '7.png', '8.png']:
    f = pathlib.Path(i)
    f.unlink()

if __name__ == "__main__":
    merge_image_colab(1)