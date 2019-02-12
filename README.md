# PaperHelicopterInvert

A code for producing inverse kinematics of a paper helicopter from iPhone data.

## Background

This data was taken from a SEB113 class that was exploring how wing dimensions influence the time of flight. I was a TA (teaching assistant) for the class and took some videos of the helicopter flights using my iPhone.

In helicopter_detector.py I try to identify the wings of the helicopters by doing some graph-based (RAG) clustering and then dilating and contracting the classes. This is somewhat successful, however more work needs to be done.
