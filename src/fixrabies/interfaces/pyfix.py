from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    Directory,
    File,
    InputMultiPath,
    TraitedSpec,
    traits,
)


class TrainingInputSpec(CommandLineInputSpec):
    species = traits.Str(
        argstr="-s %s",
        mandatory=True,
        position=1,
    )
    model_stem = traits.Str(
        argstr="%s",
        mandatory=True,
        position=2,
    )
    mel_icas = InputMultiPath(
        Directory(exists=True),
        argstr="%s",
        mandatory=True,
        position=-1,
    )


class TrainingOutputSpec(TraitedSpec):
    model_file = File(exists=True)


class Training(CommandLine):
    _cmd = "fix -t"
    input_spec = TrainingInputSpec
    output_spec = TrainingOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["model_file"] = f"{self.inputs.model_stem}.pyfix_model"
        return outputs
