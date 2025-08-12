from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy

from promptings.CoEvolve import CoEvolve 
from promptings.CoEvolvev2 import CoEvolvev2
from promptings.CoEvolvev4 import CoEvolvev4
from promptings.CoEvolvev5 import CoEvolvev5
from promptings.MapCoder import MapCoder


class PromptingFactory:
    @staticmethod
    def get_prompting_class(prompting_name):
        if prompting_name == "CoT":
            return CoTStrategy
        elif prompting_name == "MapCoder":
            return MapCoder
        elif prompting_name == "CoEvolve":
            return CoEvolve
        elif prompting_name == "Direct":
            return DirectStrategy
        elif prompting_name == "Analogical":
            return AnalogicalStrategy
        elif prompting_name == "SelfPlanning":
            return SelfPlanningStrategy
        elif prompting_name == "CoEvolvev2":
            return CoEvolvev2
        elif prompting_name == "CoEvolvev4":
            return CoEvolvev4
        elif prompting_name == "CoEvolvev5":
            return CoEvolvev5

        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
