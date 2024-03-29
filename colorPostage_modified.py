#!/bin/env python3
'''
> head coords.txt
# ra         dec             outfile(optional)
33.995270    -5.011639       a.png
33.994442    -4.996707       b.png
33.994669    -5.001553       c.png
33.996395    -5.008107       d.png
33.995679    -4.993945       e.png
33.997352    -5.010902       f.png
33.997315    -5.012523       g.png
33.997438    -5.011647       h.png
33.997379    -5.010878       i.png
33.996636    -5.008742       j.png
> python colorPostage.py --user YOUR_ACCOUNT --outDir pngs coords.txt  

'''

import argparse
import tarfile
import subprocess
import tempfile
import getpass
import os, os.path
import contextlib
import logging ; logging.basicConfig(level=logging.INFO)
try:
    import pyfits
except:
    import astropy.io.fits as pyfits
import numpy
import PIL.Image



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outDir', '-o', required=True)
    parser.add_argument('--user', '-u', required=True)
    parser.add_argument('--password', '-p', required=True)
    parser.add_argument('--filters', '-f', nargs=3, default=['HSC-I', 'HSC-R', 'HSC-G'])
    #parser.add_argument('--fov', default='30asec')
    parser.add_argument('--rerun', default='any', choices='any pdr3_dud pdr3_wide'.split())
    parser.add_argument('--color', choices='hsc sdss'.split(), default='hsc')
    parser.add_argument('--desaturate', choices='true false'.split(), default='true')
    parser.add_argument('--input', type=argparse.FileType('r'))
    args = parser.parse_args()

    password = args.password
    checkPassword(args.user, password)

    coords, outs, fovs = loadCoords(args.input)
#     print(coords)
    do_desaturate = args.desaturate
    mkdir_p(args.outDir)

    batchSize = 5
    for batchI, (batchCoords, batchFovs) in enumerate(zip(batch(coords, batchSize), batch(fovs,batchSize))):
        with requestFileFor(batchCoords, args.filters, batchFovs, args.rerun) as requestFile:
            try:
                tarMembers = queryTar(args.user, password, requestFile)
                for i, rgb in rgbBundle(tarMembers):
                    j = batchI * batchSize + i
                    outFile = os.path.join(args.outDir, outs[j])
                    logging.info('-> {}'.format(outFile))
#                     print(rgb)
                    makeColorPng(rgb, outFile, args.color, do_desaturate)
            except:
                print('Something went wrong with {}'.format(batchI))
                


TOP_PAGE = 'https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/'
API = 'https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout'


def loadCoords(input):
    import re
    comment = re.compile('\s*(?:$|#)')
    num = 1
    coords = []
    outs = []
    fovs=[]
    for line in input:
        if comment.match(line):
            continue
        cols = line.split()
        if len(cols) == 2:
            ra, dec = cols
            out = '{}.png'.format(num)
        else:
            ra, dec, out, fov = cols
        ra = float(ra)
        dec = float(dec)
        num += 1
        coords.append([ra, dec])
        outs.append(out)
        fovs.append('{}asec'.format(numpy.round(float(fov),1)))
    return coords, outs, fovs


@contextlib.contextmanager
def requestFileFor(coords, filters, fovs, rerun):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write('#? filter ra dec sw sh rerun\n'.encode('utf-8'))
        for coord, fov in zip(coords, fovs):
            for filterName in filters:
                tmp.write('{} {} {} {} {} {}\n'.format(filterName, coord[0], coord[1], fov, fov, rerun).encode())
            
        tmp.flush()
        yield tmp.name


def batch(arr, n):
    i = 0
    while i < len(arr):
        yield arr[i : i + n]
        i += n


def rgbBundle(files):
    mktmp = tempfile.NamedTemporaryFile
    with mktmp() as r, mktmp() as g, mktmp() as b:
        lastObjNum = 0
        rgb = {}
        for info, fileObj in files:
            resNum = int(os.path.basename(info.name).split('-')[0])
            objNum = (resNum - 2) // 3
            if lastObjNum != objNum:
                yield lastObjNum, rgb
                rgb.clear()
                lastObjNum = objNum
            ch = 'gbr'[resNum % 3]
            dst = locals()[ch]
            try:
                copyFileObj(fileObj, dst)
                rgb[ch] = dst.name
            except:
                print('Something went wrong with {}'.format(info.name))
            
        yield objNum, rgb



def copyFileObj(src, dst):
    dst.seek(0)
    dst.write(src.read())
    dst.truncate()


def checkPassword(user, password):
    with tempfile.NamedTemporaryFile() as netrc:
        netrc.write('machine hsc-release.mtk.nao.ac.jp login {} password {}\n'.format(user, password).encode('ascii'))
        netrc.flush()
        httpCode = subprocess.check_output(['curl', '--netrc-file', netrc.name, '-o', os.devnull, '-w', '%{http_code}', '-s', TOP_PAGE]).strip()
        if httpCode == b'401':
            raise RuntimeError('Account or Password is not correct')


def queryTar(user, password, requestFile):
    with tempfile.NamedTemporaryFile() as netrc:
        netrc.write('machine hsc-release.mtk.nao.ac.jp login {} password {}\n'.format(user, password).encode('ascii'))
        netrc.flush()
        pipe = subprocess.Popen([
            'curl', '--netrc-file', netrc.name,
            '--form', 'list=@{}'.format(requestFile),
            '--silent',
            API,
        ], stdout=subprocess.PIPE)

        with tarfile.open(fileobj=pipe.stdout, mode='r|*') as tar:
            while True:
                info = tar.next()
                if info is None: break
                logging.info('extracting {}...'.format(info.name))
                f = tar.extractfile(info)
                yield info, f
                f.close()

        pipe.wait()


def makeColorPng(rgb, out, color, do_desaturate):
    if len(rgb) == 0:
        return

    with pyfits.open(list(rgb.values())[0]) as hdul:
        template = hdul[1].data

    layers = [numpy.zeros_like(template) for i in range(3)]
    for i, ch in enumerate('rgb'):
        if ch in rgb:
            with pyfits.open(rgb[ch]) as hdul:
                x = scale(hdul[1].data, hdul[0].header['FLUXMAG0'])
                layers[i] = x

    if color == 'hsc':
        layers = hscColor(layers)
    elif color == 'sdss':
        layers = sdssColor(layers)
    else:
        assert False

    layers = numpy.array(layers)
    layers[layers < 0] = 0
    layers[layers > 1] = 1
    layers = layers.transpose((1, 2, 0))[::-1, :, :]
    
    if do_desaturate == 'true':
        desaturated_layers = desaturate_rgb(layers)
        desaturated_layers = numpy.array(255 * desaturated_layers, dtype=numpy.uint8)
    elif do_desaturate == 'false':
        desaturated_layers = layers
        desaturated_layers = numpy.array(255 * desaturated_layers, dtype=numpy.uint8)
    #desaturated_layers = desaturate_rgb(layers)
    img = PIL.Image.fromarray(desaturated_layers)
    img_resized = img.resize((424,424))
    img_resized.save(out)
    #img.save(out)



def desaturate_rgb(rgb):
    RGBim = numpy.array([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])
    a = RGBim.mean(axis=0)  # a is mean pixel value across all bands, (h, w) shape
    h, w = a.shape
    # putmask: given array and mask, set all mask=True values of array to new value
    numpy.putmask(a, a == 0.0, 1.0)  # set pixels with 0 mean value to mean of 1. Inplace?
    acube = numpy.resize(a, (3, h, w))  # copy mean value array (h,w) into 3 bands (3, h, w)
    bcube = (RGBim / acube) / 2.5  # bcube: divide image by mean-across-bands pixel value, and again by 2.5 (why?)
    mask = numpy.array(bcube)  # isn't bcube already an array?
    wt = numpy.max(mask, axis=0)  # maximum per pixel across bands of mean-band-normalised rescaled image
    # i.e largest relative deviation from mean
    numpy.putmask(wt, wt > 1.0, 1.0)  # clip largest allowed relative deviation to one (inplace?)
    wt = 1 - wt  # invert relative deviations
    wt = numpy.sin(wt*numpy.pi/2.0)  # non-linear rescaling of relative deviations
    temp = RGBim * wt + a*(1-wt) + a*(1-wt)**2 * RGBim  # multiply by weights in complicated fashion
    rgb = numpy.zeros((h, w, 3), numpy.float32)  # reset rgb to be blank
    for idx, im in enumerate((temp[0, :, :], temp[1, :, :], temp[2, :, :])):  # fill rgb with weight-rescaled rgb
        rgb[:, :, idx] = im

    clipped = numpy.clip(rgb, 0., 1.)
    return clipped

def scale(x, fluxMag0):
    mag0 = 19
    scale = 10 ** (0.4 * mag0) / fluxMag0
    x *= scale
    return x


def hscColor(rgb):
    u_min = -0.05
    u_max = 2. / 3.
    u_a = numpy.exp(10.)
    for i, x in enumerate(rgb):
        x = numpy.arcsinh(u_a*x) / numpy.arcsinh(u_a)
        x = (x - u_min) / (u_max - u_min)
        rgb[i] = x
    return rgb


def sdssColor(rgb):
    u_a = numpy.exp(10.)
    u_b = 0.05
    r, g, b = rgb
    I = (r + g + b) / 3.
    for i, x in enumerate(rgb):
        x = numpy.arcsinh(u_a * I) / numpy.arcsinh(u_a) / I * x
        x += u_b
        rgb[i] = x
    return rgb


def mkdir_p(d):
    try:
        os.makedirs(d)
    except:
        pass


if __name__ == '__main__':
    main()
