[app]
# (str) Title of your application
title = Anonymity

# (str) Package name
package.name = anonymity

# (str) Package domain (needed for android packaging)
package.domain = org.vloggerguard

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,onnx,html,js,css,json,npy

# (list) List of inclusions using pattern matching
#source.include_patterns = assets/*,images/*.png

# (list) Source files to exclude (let empty to include all the files)
source.exclude_exts = spec,avi,mp4,wav,md,txt,log

# (list) List of directory to exclude (let empty to include all the files)
source.exclude_dirs = tests, bin, venv, .venv, .git, .agent, anonymity_ui/node_modules, anonymity_ui/src, recordings, logs, output

# (list) List of exclusions using pattern matching
#source.exclude_patterns = license,images/*/*

# (str) Application versioning (method 1)
version = 0.1

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3, kivy, opencv, numpy

# (str) Custom source folders for requirements
# packagelist.vendor.dir = ../vendor

# (list) Garden requirements
#garden_requirements =

# (str) Presplash of the application
#presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/data/icon.png

# (str) Supported orientations (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (list) List of service to declare
#services = NAME:ENTRYPOINT_TO_PY,NAME2:ENTRYPOINT_TO_PY

#
# Android specific
#

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 1

# (list) Permissions
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, RECORD_AUDIO

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK will support.
android.minapi = 21

# (int) Android SDK version to use
android.sdk = 33

# (str) Android NDK version to use
#android.ndk = 25b

# (bool) Use --private data storage (True) or --dir public storage (False)
#android.private_storage = True

# (str) Android NDK directory (if empty, it will be automatically downloaded.)
#android.ndk_path =

# (str) Android SDK directory (if empty, it will be automatically downloaded.)
#android.sdk_path =

# (str) ANT directory (if empty, it will be automatically downloaded.)
#android.ant_path =

# (bool) If True, then skip trying to update the Android sdk
# This can be useful to avoid excess downloads or network errors
#android.skip_update = False

# (bool) If True, then automatically accept SDK license
# agreements. This is intended for automation only. If set to False,
# the default, you will be shown the license when installing SDK
android.accept_sdk_license = True

# (str) Android entry point, default is to use main.py
android.entrypoint = main_android.py

# (list) Android additionnal libraries to copy into libs/armeabi
#android.add_libs_armeabi = libs/android-v7/libgnustl_shared.so

# (str) python-for-android branch to use, defaults to master
p4a.branch = 2023.09.16

# (str) OUYA Console category. Should be one of GAME or APP
# If you leave this blank, OUYA support will not be enabled
#android.ouya.category = APP

# (str) Filename of movie to run at launch (full path)
#android.meta_data =

# (list) Android application meta-data to set (key=value format)
#android.meta_data =

# (list) Android library project to add (optional)
#android.library_references =

# (list) Android shared libraries which will be added to AndroidManifest.xml
#android.uses_library =

# (str) Android logcat filters to use
#android.logcat_filters = *:S python:D

# (str) Android additional Java classes to add to the project.
#android.add_javaclasses = projects/classes

# (list) The Android archs to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.archs = arm64-v8a, armeabi-v7a

# (bool) enables Android auto backup feature (optional)
android.allow_backup = True

# (list) list of Java .jar files to add to the libs dir
#android.add_jars = foo.jar,bar.jar,path/to/baz.jar

# (list) list of Java files to add to the project (can be java or a directory containing the files)
#android.add_src =

# (list) Android AAR archives to add
#android.add_aars =

# (list) Gradle dependencies
#android.gradle_dependencies =

# (list) add java compile options
# this can for example be used to enable the desugaring jdk libs
#android.add_compile_options = "sourceCompatibility = 1.8", "targetCompatibility = 1.8"

# (list) Packaging options
#p4a.extra_args = 

# (list) External libraries to include
#p4a.local_recipes = ./recipes

# (str) Bootstrap to use for android builds
#p4a.bootstrap = sdl2

# (int) port number to open an rsync service to allow remote updates
#android.rsync_port = 873

# (list) List of types of artifact to keep (default is apk)
# buildozer.artifacts = apk, aab

[buildozer]

# (int) log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) display warning if buildozer is run as root (0 = false, 1 = true)
warn_on_root = 1

# (str) Path to build artifacts storage, relative to the main directory
# build_dir = ./.buildozer

# (str) Path to bin directory, relative to the main directory
# bin_dir = ./bin
