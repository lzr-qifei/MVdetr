import pandas as pd

# MOT metrics results
results = {
    'track/idf1': 68.46071044133477,
    'track/idp': 73.78190255220419,
    'track/idr': 63.85542168674698,
    'track/recall': 86.54618473895582,
    'track/precision': 1.3157894736842104,
    'track/num_unique_objects': 76.0,
    'track/mostly_tracked': 78.94736842105263,
    'track/partially_tracked': 18.421052631578945,
    'track/mostly_lost': 2.631578947368421,
    'track/num_false_positives': 0.0,
    'track/num_misses': 201.0,
    'track/num_switches': 144.0,
    'track/num_fragmentations': 120.0,
    'track/mota': 76.90763052208835,
    'track/motp': 65.51885869198532,
    'track/num_transfer': 55.0,
    'track/num_ascend': 89.0,
    'track/num_migrate': 7.0
}

# Convert the results to a DataFrame
df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])

# Save the DataFrame to a CSV file
csv_file_path = '/home/SENSETIME/lizirui/utils/mot_metrics_results.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved at {csv_file_path}")
