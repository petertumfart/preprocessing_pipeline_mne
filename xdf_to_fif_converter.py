import numpy as np
import pandas as pd
import pyxdf
import os
import mne

def xdf_to_fif(src, dst, sbj, store=True):
    """
    Convert .xdf files to .fif files for a given subject.

    :param src: str, path of the folder containing .xdf files.
    :param dst: str, path of the folder where to save the .fif files.
    :param sbj: str, subject identifier.
    """

    # Get filenames for the subject
    file_names = [f for f in os.listdir(src) if (sbj in f) and ('.xdf' in f)]

    for i, f_name in enumerate(file_names):
        print(f'#', end=' ')
        file = src + '/' + f_name

        # Read the raw stream:
        streams, header = pyxdf.load_xdf(file)

        # Split the streams:
        eeg_stream, marker_stream = _split_streams(streams)

        # Replace markers by cleaned markers if the subject is A03:
        if (sbj == 'A03') and ('paradigm' in f_name):
            marker_stream = _replace_markers(marker_stream, file)

        # Get the eeg data:
        eeg, eeg_ts = _extract_eeg(eeg_stream, kick_last_ch=True)
        #max_eeg_ts.append(eeg_ts.max())

        # Extract all infos from the EEG stream:
        fs, ch_names, ch_labels, eff_fs = _extract_eeg_infos(eeg_stream)

        # Extract the triggers from the marker stream:
        if 'paradigm' in f_name:
            triggers = _extract_annotations(marker_stream, first_samp=eeg_ts[0], paradigm='paradigm')
        else:
            triggers = _extract_annotations(marker_stream, first_samp=eeg_ts[0], paradigm='eye')

        # Define MNE annotations
        annotations = mne.Annotations(triggers['onsets'], triggers['duration'], triggers['description'], orig_time=None)

        # Create mne info:
        # TODO: Check what info can be added to the stream:
        info = mne.create_info(ch_names, fs, ch_labels)

        # Create the raw array and add info, montage and annotations:
        raw = mne.io.RawArray(eeg, info, first_samp=eeg_ts[0])
        raw.set_montage('standard_1005')
        raw.set_annotations(annotations)

        # Store the raw file:
        if store:
            store_name = dst + '/' + f_name[:-4] + '_raw.fif'
            raw.save(store_name, overwrite=True)


def _split_streams(streams):
    """Seperate the streams from the xdf file into an EEG-stream and a Markers stream.

    Args:
        streams (list): List with len=2 that contains the eeg and the marker streams.

    Returns:
        eeg_stream: Stream containing the EEG data.
        marker_stream: Stream containing the Markers.
    """

    assert len(streams) == 2, f'Length should be 2, got {len(streams)}'

    for s in streams:
        if s['info']['type'][0] == 'EEG':
            eeg_stream = s
        elif s['info']['type'][0] == 'Marker':
            marker_stream = s

    return eeg_stream, marker_stream


def _replace_markers(m_stream, file):
    fname = file[:-4] + '_cleaned.csv'
    df_cleaned = pd.read_csv(fname)

    time_series_cleaned = np.array(df_cleaned.time_series).tolist()
    time_series_cleaned = [[mark] for mark in time_series_cleaned]

    time_stamps_cleaned = np.array(df_cleaned.time_stamps)

    m_stream['time_series'] = time_series_cleaned
    m_stream['time_stamps'] = time_stamps_cleaned

    return m_stream

def _extract_eeg(stream, kick_last_ch=True):
    """
    Extracts the EEG data and the EEG timestamp data from the stream and stores it into two lists.
    :param stream: XDF stream containing the EEG data.
    :param kick_last_ch: Boolean to kick out the brainproducts marker channel
    :return: eeg: list containing the eeg data
             eeg_ts: list containing the eeg timestamps.cd
    """
    extr_eeg = stream['time_series'].T
    extr_eeg *= 1e-6 # Convert to volts.
    assert extr_eeg.shape[0] == 65
    extr_eeg_ts = stream['time_stamps']

    if kick_last_ch:
        # Kick the last row (unused Brainproduct markers):
        extr_eeg = extr_eeg[:64,:]

    return extr_eeg, extr_eeg_ts

def _extract_eeg_infos(stream):
    """
    Takes eeg stream and extracts the sampling rate, channel names, channel labels and the effective sample rate from the xdf info.
    :param stream: EEG xdf stream
    :return: sampling_rate: Configured sampling rate
    :return: names: channel names
    :return: labels: channel labels (eeg or eog)
    :return: effective_sample_frequency: Actual sampling frequency based on timestamps.
    """
    # Extract all infos from the EEG stream:
    recording_device = stream['info']['name'][0]
    sampling_rate = float(stream['info']['nominal_srate'][0])
    effective_sample_frequency = float(stream['info']['effective_srate'])

    # Extract channel names:
    chn_names = [stream['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(64)]
    # chn_names.append('Markers')
    labels = ['eeg' for i in range(64)]
    labels[16] = 'eog'
    labels[21] = 'eog'
    labels[40] = 'eog'
    # chn_labels.append('misc')

    return sampling_rate, chn_names, labels, effective_sample_frequency


def _extract_annotations(mark_stream, first_samp, paradigm):
    """
    Function to extract the triggers of the marker stream in order to prepare for the annotations.
    :param mark_stream: xdf stream containing the markers and time_stamps
    :param first_samp: First EEG sample, serves for aligning the markers
    :return: triggs: Dict containing the extracted triggers.
    """
    triggs = {'onsets': [], 'duration': [], 'description': []}

    # Extract the markers:
    marks = mark_stream['time_series']

    # Fix markers due to bug in paradigm only if paradigm=='paradigm':
    if paradigm == 'paradigm':
        corrected_markers = _fix_markers(marks)
    else:
        corrected_markers = marks

    # Extract the timestamp of the markers and correct them to zero
    marks_ts = mark_stream['time_stamps'] - first_samp

    # Read every trigger in the stream
    for index, marker_data in enumerate(corrected_markers):
        # extract triggers information
        triggs['onsets'].append(marks_ts[index])
        triggs['duration'].append(int(0))
        # print(marker_data[0])
        triggs['description'].append(marker_data[0])

    return triggs

def _fix_markers(orig_markers):
    """
    Given a list of markers, this function processes the markers and modifies the trial type markers if necessary.
    Due to shuffle-bug in the paradigm.

    :param orig_markers: A list of markers. Each marker is a tuple containing the marker string and a float value representing the time at which the marker occurred.
    :type orig_markers: list
    :return: The modified list of markers.
    :rtype: list
    """

    trial_type_markers = ['LTR-s', 'LTR-l','RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
    counter_letter = {'l': 'R', 'r': 'L', 'b': 'T', 't': 'B'}

    # Parse through markers
    for i in range(len(orig_markers)-3):
        marker = orig_markers[i][0]
        if marker in trial_type_markers:
            following_markers = []
            # Find the next 4 occurances that start with 'c':
            # and store them in a list:
            if (i+9) < len(orig_markers):
                for ii in range(i+1, i+9):
                    next_mark = orig_markers[ii][0]
                    if next_mark[0] == 'c':
                        following_markers.append(next_mark[2])
            else:
                for ii in range(i+1, len(orig_markers)):
                    next_mark = orig_markers[ii][0]
                    if next_mark[0] == 'c':
                        following_markers.append(next_mark[2])

            # Exit loop if less than 4 following markers were found:
            if len(following_markers) < 4:
                continue

            if following_markers[0] == 'c' or following_markers[1] == 'c':
                continue

            # Extract first letter of the trial type marker:
            first_letter = marker[0].lower()
            last_letter = marker[-1].lower()

            # Check if the first two letters in following markers are the same, if not, change type:
            if (following_markers[0] != first_letter) and (following_markers[1] != first_letter):
                # Trial type changes:
                new_type = following_markers[0].upper() + 'T' + counter_letter[following_markers[0]] + '-'

                if (following_markers[2] == 'c') and (following_markers[3] == 'c'):
                    new_type = new_type + 's'
                else:
                    new_type = new_type + 'l'

                orig_markers[i][0] = new_type

            # Otherwise check if the second two markers are short or long and change accordingly:
            else:
                if (last_letter == 's') and (following_markers[2] != 'c') and (following_markers[3] != 'c'):
                    new_type = marker[:-1]
                    new_type += 'l'
                    orig_markers[i][0] = new_type

                elif (last_letter == 'l') and (following_markers[2] == 'c') and (following_markers[3] == 'c'):
                    new_type = marker[:-1]
                    new_type += 's'
                    orig_markers[i][0] = new_type


    return orig_markers


