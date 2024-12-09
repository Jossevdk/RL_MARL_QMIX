import cProfile
import pstats
import io

# Profile the train_qmix function and save the results to a file
profile_filename = 'train_qmix_profile.prof'

# Load the profiling results
profile_stats = pstats.Stats('test.prof')

# Create a StringIO object to capture the output
output = io.StringIO()

# Redirect the output to the StringIO object
profile_stats.sort_stats('cumtime').stream = output
profile_stats.print_stats()

# Get the string from the StringIO object
profiling_results = output.getvalue()

# Write the profiling results to a text file
with open("profiling_results.txt", "w") as f:
    f.write(profiling_results)