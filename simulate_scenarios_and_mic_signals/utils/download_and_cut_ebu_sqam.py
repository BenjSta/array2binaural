
if __name__ == '__main__':
    import requests, zipfile, io, os, soundfile, shutil
    dirpath = os.path.dirname(os.path.abspath(__file__))
    print(' ')
    print('USER INPUT REQUIRED')
    print(' ')
    print('This script downloads the EBU SQAM audio files from ' +
          'https://tech.ebu.ch/files/live/sites/tech/files/shared/testmaterial/SQAM_FLAC.zip. ' +
          'Only proceed if you have the rights to download these audio files. Confirm with "y" to proceed.')
    i = input()
    assert i.lower() == 'y', 'Download was not confirmed.'
    
    my_tmpdir = 'tmp_download_and_cut_sqam'
    os.mkdir(my_tmpdir)
    print('Downloading EBU SQAM audio files ...')
    r = requests.get('https://tech.ebu.ch/files/live/sites/tech/files/shared/testmaterial/SQAM_FLAC.zip')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(my_tmpdir)

    FS = 44100

    start_time_english = int(0.5 * FS)
    end_time_english = int(3.0 * FS)
    s, fs_sig = soundfile.read(os.path.join(my_tmpdir, '50.flac'))
    assert fs_sig == FS
    s = s[start_time_english:end_time_english, 0]
    soundfile.write(os.path.join(dirpath, '../source_audio/ebu_sqam/50_cut.wav'), s, FS)

    s, fs_sig = soundfile.read(os.path.join(my_tmpdir, '49.flac'))
    assert fs_sig == FS
    s = s[start_time_english:end_time_english, 0]
    soundfile.write(os.path.join(dirpath, '../source_audio/ebu_sqam/49_cut.wav'), s, FS)

    
    start_time_german_french = int(0.2 * FS)
    end_time_german_french = int(2.7 * FS)

    s, fs_sig = soundfile.read(os.path.join(my_tmpdir, '52.flac'))
    assert fs_sig == FS
    s = s[start_time_german_french:end_time_german_french, 0]
    soundfile.write(os.path.join(dirpath, '../source_audio/ebu_sqam/52_cut.wav'), s, FS)

    s, fs_sig = soundfile.read(os.path.join(my_tmpdir, '53.flac'))
    assert fs_sig == FS
    s = s[start_time_german_french:end_time_german_french, 0]
    soundfile.write(os.path.join(dirpath, '../source_audio/ebu_sqam/53_cut.wav'), s, FS)

    print('Done with downloading and cutting.')
    shutil.rmtree(my_tmpdir)
