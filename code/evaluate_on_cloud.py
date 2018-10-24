import os
import zipfile
import argparse

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

def zip_directories(output_directory):
    folders = [os.path.join(output_directory, x) for x in os.listdir(output_directory)]
    for folder in folders:
        zip_directory(folder, folder + '.zip')

def upload_zips_to_cloud(output_directory, provider):
    if provider == 'aws':
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory")
    parser.add_argument("--provider", default='aws')
    parser.add_argument("--permute", action='store_true')
    parser.add_argument("--bucket", default='bsseval')
    parser.add_argument("--upload_folder", default='uploads')
    args = parser.parse_args()

    zip_directories(args.output_directory)
