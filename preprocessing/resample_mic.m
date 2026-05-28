clc;
clear all;
list_data = dir(fullfile('*gain_1_dacq_COM03*.dat'));

for idx = 1:length(list_data)
    filename = list_data(idx).name;

    [h, x_volts_4_channel, x_bits] = processFile(filename);   % US25
    x_volts = x_volts_4_channel(1,:);

    Fs = 25E3;  % Sampling rate
    start_sample = 15 * Fs + 1;
    end_sample = 60 * Fs;
    samplesec = 2;      % Duration of each segment in seconds
    samples_per_section = Fs * samplesec;
    
    % Trim the time series
    x_volts = x_volts(start_sample:end_sample);
    
    % plot trimmed signal
    figure;
    plot(x_volts);
    title(['Trimmed Time Series for ', filename]);
    xlabel('Sample Index');
    ylabel('Amplitude (Volts)');

    % Split filename for naming
    str_arr = strsplit(filename, '_');
    label = strsplit(str_arr{end}, '.');
    base_name = "TimeSeriesMic_" + str_arr{end-1} + "_" + string(label(1));

    % Calculate number of chunks
    num_sample = floor(length(x_volts) / samples_per_section);

    for sample_idx = 1:num_sample
        sample_start = (sample_idx - 1) * samples_per_section + 1;
       sample_end = sample_idx * samples_per_section;
        x_chunk = x_volts(sample_start:sample_end);

        % Save each chunk separately
        savename = base_name + "_chunk" + num2str(sample_idx, '%02d') + ".mat";
        save(savename, 'x_chunk');
        disp(['Saved: ', savename]);
    end
end