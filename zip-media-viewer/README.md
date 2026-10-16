# ZIP Media Viewer (Android)

Android ZIP photo/video browser. The original ZIP is copied into private cache for random access; individual images are decoded on demand and videos are extracted to temporary cache before playback. The archive is **not fully extracted**. Android's native VideoView supports only device-supported codecs.

## APK
Open the repository **Actions** tab → **Build ZIP Media Viewer APK** → latest successful run → download `ZIPMediaViewer-debug-apk` artifact, unzip artifact and install `app-debug.apk`.

Android source is isolated in `zip-media-viewer/`; existing CoreGrid files are not modified.
