## Overview 

A [Workflow](../../reference/griptape/structures/workflow.md) is a non-sequential DAG that can be used for complex concurrent scenarios with tasks having multiple inputs.


## Context

Workflows have access to the following [context](../../reference/griptape/structures/workflow.md#griptape.structures.workflow.Workflow.context) variables in addition to the [base context](./tasks.md#context):

* `parent_outputs`: outputs into the current task referenceable by parent task IDs.
* `parents`: parent tasks referenceable by IDs.
* `children`: child tasks referenceable by IDs.

## Workflow
Let's build a simple workflow. Let's say, we want to write a story in a fantasy world with some unique characters. We could setup a workflow that generates a world based on some keywords. Then we pass the world description to any number of child tasks that create characters. Finally, the last task pulls in information from all parent tasks and writes up a short story.

```python
from griptape.tasks import PromptTask
from griptape.structures import Workflow


world_task = PromptTask(
    "Create a fictional world based on the following key words {{ keywords|join(', ') }}",
    context={
        "keywords": ["fantasy", "ocean", "tidal lock"]
    },
    id="world"
)

def character_task(task_id, character_name) -> PromptTask:
    return PromptTask(
        "Based on the following world description create a character named {{ name }}:\n{{ parent_outputs['world'] }}",
        context={
            "name": character_name
        },
        id=task_id,
        parent_ids=["world"]
    )

scotty_task = character_task("scotty", "Scotty")
annie_task = character_task("annie", "Annie")

story_task = PromptTask(
    "Based on the following description of the world and characters, write a short story:\n{{ parent_outputs['world'] }}\n{{ parent_outputs['scotty'] }}\n{{ parent_outputs['annie'] }}",
    id="story",
    parent_ids=["world", "scotty", "annie"]
)

workflow = Workflow(tasks=[world_task, story_task, scotty_task, annie_task, story_task])

workflow.run()
```


!!! Info
    Output edited for brevity
```
[09/08/23 10:26:21] INFO     PromptTask world
                             Input: Create a fictional world based on the following key words fantasy, ocean, tidal lock
[09/08/23 10:27:11] INFO     PromptTask world
                             Output: Welcome to the world of "Oceanus Fantasia", a realm where fantasy and oceanic wonders intertwine in a
                             unique celestial phenomenon known as tidal lock.

                             Oceanus Fantasia is a vast, water-dominated planet, with its surface covered by a seemingly endless ocean. The
                             ocean is not just water, but a magical liquid that sparkles with all the colors of the rainbow under the planet's
                             twin suns. This magical ocean is home to a myriad of fantastical creatures, from the tiny luminescent pixie-fish
                             to the colossal, wise leviathans that roam the depths.

                             ...

                             In Oceanus Fantasia, magic and technology coexist. The inhabitants use their understanding of the magical ocean
                             and its creatures to power their cities, heal their sick, and even control the weather. The ocean's magic is a
                             part of them, and they are a part of the ocean.

                             Welcome to Oceanus Fantasia, where the ocean's tide writes the story of a world in perpetual twilight, a world of
                             magic, mystery, and endless adventure.
                    INFO     PromptTask scotty
                             Input: Based on the following world description create a character named Scotty:
                             Welcome to the world of "Oceanus Fantasia", a realm where fantasy and oceanic wonders intertwine in a unique
                             celestial phenomenon known as tidal lock.

                             ...

                             In Oceanus Fantasia, magic and technology coexist. The inhabitants use their understanding of the magical ocean
                             and its creatures to power their cities, heal their sick, and even control the weather. The ocean's magic is a
                             part of them, and they are a part of the ocean.

                             Welcome to Oceanus Fantasia, where the ocean's tide writes the story of a world in perpetual twilight, a world of
                             magic, mystery, and endless adventure.
                    INFO     PromptTask annie
                             Input: Based on the following world description create a character named Annie:
                             Welcome to the world of "Oceanus Fantasia", a realm where fantasy and oceanic wonders intertwine in a unique
                             celestial phenomenon known as tidal lock.

                             ...

                             Welcome to Oceanus Fantasia, where the ocean's tide writes the story of a world in perpetual twilight, a world of
                             magic, mystery, and endless adventure.
[09/08/23 10:27:47] INFO     PromptTask scotty
                             Output: Character Name: Scotty

                             Scotty is a Tide Whisperer, one of the mysterious seers who reside in the Twilight Zone of Oceanus Fantasia. He
                             is a tall, slender figure with a complexion that reflects the perpetual twilight of his home, a mix of the golden
                             hue of the Solaris and the luminescent glow of the Lunarians. His hair, a cascade of silver waves, mirrors the
                             ever-changing colors of the magical ocean. His eyes, a deep sea-green, hold a depth that seems to echo the
                             vastness of the ocean itself.

                             ...

                             Scotty embodies the spirit of Oceanus Fantasia, a world of magic, mystery, and endless adventure. He is a beacon
                             of hope and wisdom in a world divided by light and darkness, a testament to the unity and harmony that can exist
                             in diversity.
[09/08/23 10:28:47] INFO     PromptTask annie
                             Output: Character Name: Annie

                             Annie is a young, vibrant inhabitant of the Twilight Zone in Oceanus Fantasia. She is a Tide Whisperer, a seer
                             who can read the future in the ebb and flow of the magical tides. Her skin is a soft, iridescent hue, a blend of
                             the golden tones of the Solaris people and the luminescent glow of the Lunarians, reflecting her unique position
                             between the two cultures.

                             ...

                             Annie's best friend is a luminescent pixie-fish named Lumi. Lumi is a constant companion and often assists Annie
                             in her tide readings. Together, they represent the spirit of Oceanus Fantasia - a world of magic, mystery, and
                             endless adventure.
                    INFO     PromptTask story
                             Input: Based on the following description of the world and characters, write a short story:
                             Welcome to the world of "Oceanus Fantasia", a realm where fantasy and oceanic wonders intertwine in a unique
                             celestial phenomenon known as tidal lock.

                             ...

                             Annie's best friend is a luminescent pixie-fish named Lumi. Lumi is a constant companion and often assists Annie
                             in her tide readings. Together, they represent the spirit of Oceanus Fantasia - a world of magic, mystery, and
                             endless adventure.
[09/08/23 10:29:47] INFO     PromptTask story
                             Output: In the world of Oceanus Fantasia, where the ocean's tide writes the story of a world in perpetual
                             twilight, lived Scotty and Annie, the Tide Whisperers. They were the bridge between the Solaris Kingdom and the
                             Lunarian Clan, the mediators of the perpetual day and eternal night.
                             ...
                             Scotty and Annie, with their deep understanding of the magical ocean and its creatures, continued to guide the
                             inhabitants of Oceanus Fantasia. They embodied the spirit of their world, a world of magic, mystery, and endless
                             adventure. They were the beacon of hope and wisdom in a world divided by light and darkness, a testament to the
                             unity and harmony that can exist in diversity.
```

### Declarative vs Imperative Syntax

The above example showed how to create a workflow using the declarative syntax via the `parent_ids` init param, but there are a number of declarative and imperative options for you to choose between. There is no functional difference, they merely exist to allow you to structure your code as is most readable for your use case. Possibilities are illustrated below.

Declaratively specify parents (same as above example):

```python
from griptape.tasks import PromptTask
from griptape.structures import Workflow
from griptape.rules import Rule

workflow = Workflow(
    tasks=[
        PromptTask("Name an animal", id="animal"),
        PromptTask("Describe {{ parent_outputs['animal'] }} with an adjective", id="adjective", parent_ids=["animal"]),
        PromptTask("Name a {{ parent_outputs['adjective'] }} animal", id="new-animal", parent_ids=["adjective"]),
    ],
    rules=[Rule("output a single lowercase word")]
)

workflow.run()
```

Declaratively specify children:

```python
from griptape.tasks import PromptTask
from griptape.structures import Workflow
from griptape.rules import Rule

workflow = Workflow(
    tasks=[
        PromptTask("Name an animal", id="animal", child_ids=["adjective"]),
        PromptTask("Describe {{ parent_outputs['animal'] }} with an adjective", id="adjective", child_ids=["new-animal"]),
        PromptTask("Name a {{ parent_outputs['adjective'] }} animal", id="new-animal"),
    ],
    rules=[Rule("output a single lowercase word")],
)

workflow.run()
```

Declaratively specifying a mix of parents and children:

```python
from griptape.tasks import PromptTask
from griptape.structures import Workflow
from griptape.rules import Rule

workflow = Workflow(
    tasks=[
        PromptTask("Name an animal", id="animal"),
        PromptTask("Describe {{ parent_outputs['animal'] }} with an adjective", id="adjective", parent_ids=["animal"], child_ids=["new-animal"]),
        PromptTask("Name a {{ parent_outputs['adjective'] }} animal", id="new-animal"),
    ],
    rules=[Rule("output a single lowercase word")],
)

workflow.run()
```

Imperatively specify parents:

```python
from griptape.tasks import PromptTask
from griptape.structures import Workflow
from griptape.rules import Rule

animal_task = PromptTask("Name an animal", id="animal")
adjective_task = PromptTask("Describe {{ parent_outputs['animal'] }} with an adjective", id="adjective")
new_animal_task = PromptTask("Name a {{ parent_outputs['adjective'] }} animal", id="new-animal")

adjective_task.add_parent(animal_task)
new_animal_task.add_parent(adjective_task)

workflow = Workflow(
    tasks=[animal_task, adjective_task, new_animal_task],
    rules=[Rule("output a single lowercase word")],
)

workflow.run()
```

Imperatively specify children:

```python
from griptape.tasks import PromptTask
from griptape.structures import Workflow
from griptape.rules import Rule

animal_task = PromptTask("Name an animal", id="animal")
adjective_task = PromptTask("Describe {{ parent_outputs['animal'] }} with an adjective", id="adjective")
new_animal_task = PromptTask("Name a {{ parent_outputs['adjective'] }} animal", id="new-animal")

animal_task.add_child(adjective_task)
adjective_task.add_child(new_animal_task)

workflow = Workflow(
    tasks=[animal_task, adjective_task, new_animal_task],
    rules=[Rule("output a single lowercase word")],
)

workflow.run()
```

Imperatively specify a mix of parents and children:

```python
from griptape.tasks import PromptTask
from griptape.structures import Workflow
from griptape.rules import Rule

animal_task = PromptTask("Name an animal", id="animal")
adjective_task = PromptTask("Describe {{ parent_outputs['animal'] }} with an adjective", id="adjective")
new_animal_task = PromptTask("Name a {{ parent_outputs['adjective'] }} animal", id="new-animal")

adjective_task.add_parent(animal_task)
adjective_task.add_child(new_animal_task)

workflow = Workflow(
    tasks=[animal_task, adjective_task, new_animal_task],
    rules=[Rule("output a single lowercase word")],
)

workflow.run()
```

Or even mix imperative and declarative:

```python
from griptape.tasks import PromptTask
from griptape.structures import Workflow
from griptape.rules import Rule

animal_task = PromptTask("Name an animal", id="animal")
adjective_task = PromptTask("Describe {{ parent_outputs['animal'] }} with an adjective", id="adjective", parent_ids=["animal"])


new_animal_task = PromptTask("Name a {{ parent_outputs['adjective'] }} animal", id="new-animal")
new_animal_task.add_parent(adjective_task)

workflow = Workflow(
    tasks=[animal_task, adjective_task, new_animal_task],
    rules=[Rule("output a single lowercase word")],
)

workflow.run()
```
