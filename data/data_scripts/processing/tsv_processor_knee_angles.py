import os
import csv
import numpy as np
import pandas as pd


class TSVProcessorKnee:
    def __init__(self, input_folder, participant_id, output_folder):
        self.input_folder = input_folder
        self.output_folder = os.path.join(os.path.dirname(input_folder), output_folder)

        # Import CSV file and store each column as a separate list of strings
        with open('/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RADARTreePose/data/metadata/pose_vectors.csv') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Get header row
            data = [list(row) for row in reader]
        # Create a Pandas dataframe from the lists of strings
        self.pose_df = pd.DataFrame(data, columns=header)
        self.participant_id = str(participant_id)


    def calculate_3d_angle(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
        '''Calculate 3d angle given x,y,z data points'''
        vec1 = np.array([x1-x2, y1-y2, z1-z2])
        vec2 = np.array([x3-x2, y3-y2, z3-z2])
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    def calculate_angles(self, df):
        angles = []
        for index, row in df.iterrows():
            # Calculate angles
            # print("First Row")
            # print(row['Wrist_R_pos_X'], row['Wrist_R_pos_Y'], row['Wrist_R_pos_Z'])
            angles.append([
                # Right Elbow
                self.calculate_3d_angle(float(row['Wrist_R_pos_X']), row['Wrist_R_pos_Y'], row['Wrist_R_pos_Z'],
                                row['Elbow_R_pos_X'], row['Elbow_R_pos_Y'], row['Elbow_R_pos_Z'],
                                row['Shoulder_R_pos_X'], row['Shoulder_R_pos_Y'], row['Shoulder_R_pos_Z']),
                # Left Elbow
                self.calculate_3d_angle(row['Wrist_L_pos_X'], row['Wrist_L_pos_Y'], row['Wrist_L_pos_Z'],
                                row['Elbow_L_pos_X'], row['Elbow_L_pos_Y'], row['Elbow_L_pos_Z'],
                                row['Shoulder_L_pos_X'], row['Shoulder_L_pos_Y'], row['Shoulder_L_pos_Z']),
                # Right Knee
                self.calculate_3d_angle(row['Ankle_R_pos_X'], row['Ankle_R_pos_Y'], row['Ankle_R_pos_Z'],
                                row['Knee_R_pos_X'], row['Knee_R_pos_Y'], row['Knee_R_pos_Z'],
                                row['Hip_R_Ant_pos_X'], row['Hip_R_Ant_pos_Y'], row['Hip_R_Ant_pos_Z']),
                # Left Knee
                self.calculate_3d_angle(row['Ankle_L_pos_X'], row['Ankle_L_pos_Y'], row['Ankle_L_pos_Z'],
                                row['Knee_L_pos_X'], row['Knee_L_pos_Y'], row['Knee_L_pos_Z'],
                                row['Hip_L_Ant_pos_X'], row['Hip_L_Ant_pos_Y'], row['Hip_L_Ant_pos_Z']),
                # Right Shoulder
                self.calculate_3d_angle(row['Elbow_R_pos_X'], row['Elbow_R_pos_Y'], row['Elbow_R_pos_Z'],
                            row['Shoulder_R_pos_X'], row['Shoulder_R_pos_Y'], row['Shoulder_R_pos_Z'],
                            row['Chest_pos_X'], row['Chest_pos_Y'], row['Chest_pos_Z']),
                # Left Shoulder
                self.calculate_3d_angle(row['Elbow_L_pos_X'], row['Elbow_L_pos_Y'], row['Elbow_L_pos_Z'],
                                row['Shoulder_L_pos_X'], row['Shoulder_L_pos_Y'], row['Shoulder_L_pos_Z'],
                                row['Chest_pos_X'], row['Chest_pos_Y'], row['Chest_pos_Z']),
                # Right Hip 
                self.calculate_3d_angle(row['Hip_R_Post_pos_X'], row['Hip_R_Post_pos_Y'], row['Hip_R_Post_pos_Z'],
                                row['Hip_R_Ant_pos_X'], row['Hip_R_Ant_pos_Y'], row['Hip_R_Ant_pos_Z'],
                                row['Knee_R_pos_X'], row['Knee_R_pos_Y'], row['Knee_R_pos_Z']),
                # Left Hip 
                self.calculate_3d_angle(row['Hip_L_Post_pos_X'], row['Hip_L_Post_pos_Y'], row['Hip_L_Post_pos_Z'],
                                row['Hip_L_Ant_pos_X'], row['Hip_L_Ant_pos_Y'], row['Hip_L_Ant_pos_Z'],
                                row['Knee_L_pos_X'], row['Knee_L_pos_Y'], row['Knee_L_pos_Z']),
                # Front Hips to Belly
                self.calculate_3d_angle(row['Hip_R_Ant_pos_X'], row['Hip_R_Ant_pos_Y'], row['Hip_R_Ant_pos_Z'],
                                row['Belly_pos_X'], row['Belly_pos_Y'], row['Belly_pos_Z'],
                                row['Hip_L_Ant_pos_X'], row['Hip_L_Ant_pos_Y'], row['Hip_L_Ant_pos_Z']),
                # Back Hips to Low Back
                self.calculate_3d_angle(row['Hip_R_Post_pos_X'], row['Hip_R_Post_pos_Y'], row['Hip_R_Post_pos_Z'],
                                row['Lower_Back_pos_X'], row['Lower_Back_pos_Y'], row['Lower_Back_pos_Z'],
                                row['Hip_L_Post_pos_X'], row['Hip_L_Post_pos_Y'], row['Hip_L_Post_pos_Z'],), 
                # Uppper Back to Shoulders
                self.calculate_3d_angle(row['Shoulder_L_pos_X'], row['Shoulder_L_pos_Y'], row['Shoulder_L_pos_Z'],
                                row['Upper_Back_pos_X'], row['Upper_Back_pos_Y'], row['Upper_Back_pos_Z'],
                                row['Shoulder_R_pos_X'], row['Shoulder_R_pos_Y'], row['Shoulder_R_pos_Z'],)
            ])
        
        angles_df = pd.DataFrame(angles, columns=[
            'Elbow_R_angle',
            'Elbow_L_angle',
            'Knee_R_angle',
            'Knee_L_angle',
            'Shoulder_R_angle',
            'Shoulder_L_angle',
            'Hip_R_angle',
            'Hip_L_angle',
            'Hip_Belly_Hip_angle',
            'Hip_LowBack_Hip_angle',
            'Shoulder_UpperBack_Shoulder_angle'
        ])
        
        return angles_df


    def process_files(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".tsv"):
                print(f"Working on {filename}")
                self.process_tsv(os.path.join(self.input_folder, filename))
                
    
    def add_pose_ID_column(self, df , file_path):
        file_name = os.path.basename(file_path)
        pose_column = 3
        # print(self.pose_df.columns)
        if "DOWND" in file_name:
            df.insert(pose_column, 'pose', self.pose_df['DOWND'])
        elif "CHAIR" in file_name:
            df.insert(pose_column, 'pose', self.pose_df['CHAIR'])
        elif "YOGIS" in file_name:
            df.insert(pose_column, 'pose', self.pose_df['YOGIS'])
        elif "CT2CW" in file_name:
            df.insert(pose_column, 'pose', self.pose_df['CT2CW'])
        elif "CRW2L" in file_name:
            df.insert(pose_column, 'pose', self.pose_df['CRW2L'])
        elif "CRW2R" in file_name:
            df.insert(pose_column, 'pose', self.pose_df['CRW2R'])
        elif "FF2MN" in file_name:
            df.insert(pose_column, 'pose', self.pose_df['FF2MN'])
        elif "MNTRL" in file_name:
            df.insert(pose_column, 'pose', self.pose_df['MNTRL'])
        elif "MNTRR" in file_name:
            df.insert(pose_column, 'pose', self.pose_df['MNTRR'])
        else:
            raise Exception(f"No known file type found for {self.file_name}")
        return df

    def process_tsv(self, file_path):
        with open(file_path, mode='r', newline='') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            first_5_rows_list = []
            remaining_rows_list = []

            for i, row in enumerate(tsv_reader):
                if i < 5:
                    first_5_rows_list.append(row)
                else:
                    if len(row) < 58:
                        row += [''] * (58 - len(row))
                    remaining_rows_list.append(row)

            # Extract header information
            header_info = {row[0]: row[1] for row in first_5_rows_list}

            # Create Header pandas DataFrame
            df_header = pd.DataFrame.from_dict(header_info, orient='index', columns=['Value'])
            # print("Header for data frame")
            print("df_header")
            print(df_header)
            
            # Create blank, correct shape pandas DataFrames from remainder of lists
            df = pd.DataFrame(remaining_rows_list)
            # print('New data frame')

            # Shift row 6 to the left and remove cell 6,1
            df.iloc[2, 0:-1] = df.iloc[0, 1:].values
            
            #delete empty column
            df = df.iloc[:,:-1]

            # Remove rows 7 and 8 (originally 8 and 9)
            df = df.drop(df.index[0:2])
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            # print('2: New data frame')
            # print(df)
            
            ## Change data types of columns
            df = df.apply(pd.to_numeric, downcast='float')
            
            # Add 'frame', 'time' and participant columns
            df.insert(0, 'frame', range(0, len(df)))
            df.insert(1, 'time', [i * 0.01 for i in range(len(df))])
            df.insert(2,'participant_id', self.participant_id)
            
            #Add pose ID to the data frame
            df = self.add_pose_ID_column(df,file_path)

            # Reset index
            df.reset_index(drop=True, inplace=True)
            
            ### Calculate the 3d Angles
            df_angles = self.calculate_angles(df)
            df = pd.concat([df, df_angles], axis=1)
            print(df)

            if df.shape[0] != 4000 or df.shape[1] != 72:
                print(df.shape)
                raise Exception("DATA Frame is the wrong size!!")
            else:
                # Save as CSV
                # output_file_path = os.path.join(
                #     self.output_folder, os.path.splitext(os.path.basename(file_path))[0] + ".csv"
                # )
                output_file_path = os.path.join(
                    self.output_folder, os.path.splitext(os.path.basename(file_path))[0] + ".csv"
                )
                print(f"Saved: {os.path.basename(file_path)}")
                # df.to_csv(output_file_path, index=False, header=True)
                return df

