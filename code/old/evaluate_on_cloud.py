import os
import zipfile
import argparse
from tqdm import trange
import boto3

s3 = boto3.resource('s3', region_name='us-east-1')

def zip_directory(directory, zipname):
    if os.path.exists(directory):
        outZipFile = zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED)
        rootdir = os.path.basename(directory)
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath   = os.path.join(dirpath, filename)
                parentpath = os.path.relpath(filepath, directory)
                arcname    = os.path.join(rootdir, parentpath)
                outZipFile.write(filepath, arcname)
        outZipFile.close()

def zip_directories(output_directory, bucket_name, upload_folder):
    folders = [os.path.join(output_directory, x) for x in os.listdir(output_directory) if '.zip' not in x]
    progress_bar = trange(len(folders))
    for i in progress_bar:
        folder = folders[i]
        progress_bar.set_description(folder.split('/')[-1])
        zip_directory(folder, folder + '.zip')
    #upload_to_s3(folder + '.zip', bucket_name, upload_folder)

def upload_to_s3(output_file, bucket_name, upload_folder):
    output_file_name = output_file.split('/')[-1]
    s3.meta.client.upload_file(output_file,
                               bucket_name,
                               os.path.join(upload_folder, output_file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory")
    parser.add_argument("--provider", default='aws')
    parser.add_argument("--bucket", default='bsseval')
    parser.add_argument("--upload_folder", default='uploads')
    args = parser.parse_args()

    zip_directories(args.output_directory, args.bucket, args.upload_folder)