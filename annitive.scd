///////////////////////////////////////////////////////////////////////////////////////////
//ANNitive--ANN-based additive synthesis
///////////////////////////////////////////////////////////////////////////////////////////

//TODO: selectable atsFrame, give max number as well
//TODO: spectra view
//TODO: populate with low amplitude high freq noise to provide a rich dataset for simpler spectra
//TODO: add volume control

//there are two parts to be intrepreted in the code, first place the cursor somewhere between
//the first pair of parentheses and press Cmd-Return (Ctrl-Enter on the PC)
//and then do the same with the second pair of parentheses to start the GUI
//check the post window for updates

// a monophonic clarinet sample as well as a clarinet multiphonic are added to test


//TRAINING SPECTRA
(
~analyze = Routine({
~freqsAndAmps1 = List[];
~freqsAndAmps2 = List[];
~freqsAndAmps3 = List[];
~newFreqs = Array.new;
~newAmps = Array.new;
~freqs1 = Array.new;
~amps1 = Array.new;
~freqs2 = Array.new;
~amps2 = Array.new;
~freqs3 = Array.new;
~amps3 = Array.new;

	// ~atsFile = AtsFile.new("/Users/rikyu/Desktop/PORTFOLIO/NEW_ANN/clarinet.ats".standardizePath).load;
~atsFrame1 = Array[10];
~atsFrame2 = Array[20];
~atsFrame3 = Array[15];

~freqs1.add(~atsFile.getFrameFreq(~atsFrame1).flat);
~amps1.add(~atsFile.getFrameAmp(~atsFrame1).flat);
~freqs2.add(~atsFile.getFrameFreq(~atsFrame2).flat);
~amps2.add(~atsFile.getFrameAmp(~atsFrame2).flat);
~freqs3.add(~atsFile.getFrameFreq(~atsFrame3).flat);
~amps3.add(~atsFile.getFrameAmp(~atsFrame3).flat);

~freqs1 = ~freqs1.flat;
~amps1 = ~amps1.flat;
~freqs2 = ~freqs2.flat;
~amps2 = ~amps2.flat;
~freqs3 = ~freqs3.flat;
~amps3 = ~amps3.flat;


for (0, ~freqs1[~freqs1.size-2], { arg i=0; ~freqsAndAmps1.add(~freqs1[i]); ~freqsAndAmps1.add(~amps1[i])});
~freqsAndAmps1.insert(0, ~freqs1[0]); //needed because the first index gets dropped for some reason...
for (0, ~freqs2[~freqs2.size-2], { arg i=0; ~freqsAndAmps2.add(~freqs2[i]); ~freqsAndAmps2.add(~amps2[i])});
~freqsAndAmps2.insert(0, ~freqs2[0]);
for (0, ~freqs3[~freqs3.size-2], { arg i=0; ~freqsAndAmps3.add(~freqs3[i]); ~freqsAndAmps3.add(~amps3[i])});
~freqsAndAmps3.insert(0, ~freqs3[0]);

	0.1.wait;

//write. nils will be cleared up in Python
~file1 = File("/users/rikyu/Desktop/PORTFOLIO/NEW_ANN/train_spectral_data_1.txt".standardizePath, "w");
~file1.write(~freqsAndAmps1.asCompileString); //solved the problem with "etc."
~file1.close;

~file2 = File("/users/rikyu/Desktop/PORTFOLIO/NEW_ANN/train_spectral_data_2.txt".standardizePath, "w");
~file2.write(~freqsAndAmps2.asCompileString);
~file2.close;

~file3 = File("/users/rikyu/Desktop/PORTFOLIO/NEW_ANN/test_spectral_data.txt".standardizePath, "w");
~file3.write(~freqsAndAmps3.asCompileString);
~file3.close;

	0.1.wait;

	"spectral data is ready for ANN. Head over to the Python code and start training.".postln;
});

~synthesize = Routine({
~newFreqs = Object.readArchive("/users/rikyu/Desktop/PORTFOLIO/NEW_ANN/newFreqs.txt");
~newAmps = Object.readArchive("/users/rikyu/Desktop/PORTFOLIO/NEW_ANN/newAmps.txt");

	0.1.wait;

	"spectral data is read from ANN. You can hear it now.".postln;
});
)

//GUI: first select the ATS file.
//click "Analyze for ANN" to analyze and create .txt files for the neural network training & testing
//run the python file, return to SC
//click "Synthesize for ANN" to read the generated spectrum data
//compare the source sound with the generated one by playing

(
var atsFile, enView, openButton, atsBuf, atsPath, atsDispButton, entDispButton, win, atsDisp;

//GUI
win = Window("Create training and testing spectra", Rect(300, 300, 720, 250), false).front;
win.view.background_(Color.grey);
win.alpha_(0.98);
win.onClose = {Buffer.freeAll; "Done.".postln;"".postln};


openButton = Button.new(win, Rect(290, 0, 130, 20)) //will also select the aiff file LATER
.states_([["Select ATS file", Color.black, Color.new(0.63915638923645, 0.61455166339874, 0.3189784526825)]])
.action_({
	Dialog.openPanel(
		okFunc: { |path|
			atsPath = path;
			~atsFile = AtsFile.new(path, s);
		},
		cancelFunc: {"cancelled"}
	);
});

// ANN Analysis Button
atsDispButton= Button.new(win, Rect(30, 50, 250, 30))
.states_([["Analyze for ANN",Color.black, Color.new(0.76396951675415, 0.87935035228729, 0.62494311332703)]])
.action_({
	~analyze.play;
});


// ANN Synthesis Button
~entDispButton= Button.new(win, Rect(430, 50, 250, 30))
.states_([["Synthesize from ANN",Color.black, Color.new(0.76396951675415, 0.87935035228729, 0.62494311332703)]])
.action_({
	~synthesize.play;
});

//PLAY
~playOriginal = Button.new(win, Rect(163, 150, 150, 30))
.states_([["PLAY the test spectrum",Color.black, Color.new(0.76396951675415, 0.87935035228729, 0.62494311332703)]])
.action_({
~source = SynthDef(\source, {
	var sig, env;
	sig = LeakDC.ar(Klang.ar(`[ ~freqs3, ~amps3 ],
	1, 0) * 0.4);
	env = EnvGen.kr(Env([0,1,0], [1,1]), doneAction: Done.freeSelf);
	Out.ar([0,1], sig*env);
}).play;
});

~playResult = Button.new(win, Rect(400, 150, 150, 30))
.states_([["PLAY the result",Color.black, Color.new(0.76396951675415, 0.87935035228729, 0.62494311332703)]])
.action_({

~result = SynthDef(\result, {
	var sig, env;
	sig = LeakDC.ar(Klang.ar(`[ ~newFreqs, ~newAmps ],
	1, 0) * 0.4);
	env = EnvGen.kr(Env([0,1,0], [1,1]), doneAction: Done.freeSelf);
	Out.ar([0,1], Limiter.ar(sig*env,0.1));
}).play;
});
)
