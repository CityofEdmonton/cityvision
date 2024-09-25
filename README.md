## CityVsion

CityVision is tool that allows to use traffic videos to extract information about the traffic flow in a city. The tool is able to detect vehicles, track them and count the number of vehicles that pass through a certain point. The tool is also able to detect traffic jams and to estimate the speed of the vehicles. The tool will output data in tabular format and in a graphical format.

# Dataflow (Apache Beam)

This project uses Dataflow to process the data. The dataflow is as follows:

1. Read the video file and extract the frames.
2. Process the frames to detect vehicles.
3. Track the vehicles.
4. Count the number of vehicles that pass through a certain point.
5. Aggregate data from diffrent hours (with the same camera) to get a better estimate of the traffic flow.
