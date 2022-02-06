# LionStack

A collection of deep-stack inspired utilities for recognizing Sea Lions from "in-the-wild" images and videos.
Currently in pre-alpha stages.


![image info](./lionstack-demo.gif)

To build:

If this is your first time running or you've changing anything in the video-tagging app, you'll need to rebuild the container.

```
docker-compose build lionstack_video_tagging
```

Then to run:

In one terminal:

```
docker-compose up
```

Then navigate to http://localhost:8502

Videos should be in:

```
training/videos/
```

These will be mounted into the right docker container.

