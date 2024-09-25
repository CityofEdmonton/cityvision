"""Defines a pipeline to create a banner from the longest word in the input."""

import apache_beam as beam
import pandas as pd
from cityvision import my_transforms

def vechile_counting(
    model_input: list[dict], output_table:str, pipeline_options_args: list[str]) -> beam.Pipeline:
    """Instantiates and returns a Beam pipeline object"""

    pipeline_options = beam.options.pipeline_options.PipelineOptions(
        pipeline_options_args
    )

    pipeline = beam.Pipeline(options=pipeline_options)
    with pipeline as p:
        # Step 1: Read video paths
        video_paths = (p 
                    | 'Create file input' >> beam.Create(model_input))

        # Step 2: Process each video in parallel
        vehicle_counts = (video_paths
                        | 'Process videos' >> beam.Map(my_transforms.ProcessVideo))

        # Step 3: Combine DataFrames
        combined_df = (vehicle_counts
                    | 'Combine DataFrames' >> beam.CombineGlobally(beam.combiners.ToListCombineFn())
                    | 'Filter empty DataFrames' >> beam.Filter(lambda dfs: len(dfs) > 0)
                    | 'Convert to single DataFrame' >> beam.Map(lambda dfs: pd.concat(dfs, ignore_index=True))
                    | 'Transform Dataframe' >> beam.Map(my_transforms.dataframe_transform)
                    | 'Convert to Dict' >> beam.FlatMap(my_transforms.dataframe_to_dicts))

        # step 4 : print the output
        #combined_df | 'Print Output' >> beam.Map(print)
        # Step 4: Write to BigQuery
        (combined_df
        | 'Write to BigQuery' >> my_transforms.BigQueryWriteErrorTransform(
                output_table,
                schema='uuid:STRING, Direction:STRING, timestamp:STRING, vehicle_count:INTEGER'))

    return pipeline