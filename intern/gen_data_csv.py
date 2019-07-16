#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
 
if __name__ == '__main__':
  outdir = sys.argv[1]
 
  if not os.path.isdir(outdir):
    sys.exit('%s is not directory' % outdir)
 
  names = {
    "label0": 0, # エルメス
    "label1": 1, #ヴィトン
    "label2": 2, # シャネル
    "label3": 3, # コーチ
    "label4": 4, # グッチ
    "label5": 5
    # プラダ
    #最大10個まで設定できます。画像の置いてあるフォルダーを記入してください。
  }
 
  exts = ['.JPG','.JPEG']
  print("path,value")
  for dirpath, dirnames, filenames in os.walk(outdir):
    for dirname in dirnames:
      if dirname in names:
        n = names[dirname]
        member_dir = os.path.join(dirpath, dirname)
        for dirpath2, dirnames2, filenames2 in os.walk(member_dir):
          if not dirpath2.endswith(dirname):
            continue
          for filename2 in filenames2:
            (fn,ext) = os.path.splitext(filename2)
            if ext.upper() in exts:
              img_path = os.path.join(dirpath2, filename2)
              print ('%s,%s' % (img_path, n))