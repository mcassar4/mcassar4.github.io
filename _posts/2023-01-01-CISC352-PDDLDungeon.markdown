---
layout: post  
title: "PDDL Dungeon: Treasure Hunter AI"  
date: 2023-02-26 12:00:00 +0500  
description: "A deep dive into planning and domain modeling with PDDL for a treasure hunter in a dungeon environment."  
img: ../img/pddl/pddlagent.webp
tags: [AI, PDDL, Planning, Domain Modeling]  

---

## Introduction

As part of an Artificial Intelligence project, I designed and implemented a **PDDL domain and problem set** for a treasure hunter navigating a dungeon filled with challenges and treasures. This project, completed independently, explores planning through the lens of AI, with a focus on **domain modeling, constraints, and action preconditions/effects**. 

The Treasure Hunter Domain presents a dynamic and challenging environment. The goal is for the hero to retrieve the treasure by overcoming locked and risky corridors, collecting keys, and dealing with environmental hazards. This report delves into the development process, challenges, and solutions.

---

## Role and Responsibilities
- **Domain Modeling**: Designing predicates, action preconditions, and effects to represent the dungeon's dynamics accurately.
- **Problem Instances**: Implementing multiple problem files with varying levels of complexity.
- **Helper Script**: Developing a Python script (`writer.py`) to streamline the generation of initial state descriptions and problem-specific details.

I ensured the submission met all requirements, including the design of an original "difficult" problem.

---

## Problem Overview

The domain focuses on a treasure hunter navigating a dungeon with the following features:
- **Rooms and Corridors**: Rooms are interconnected by corridors, some of which are locked or risky.
- **Keys**: Single-use, two-use, and multi-use keys unlock corridors of matching colors.
- **Environmental Hazards**: Risky corridors collapse upon use, making rooms messy with rubble.
- **Hero's Actions**: The hero can move, pick up and drop keys, unlock corridors, and clean messy rooms.

### Problem Instances

1. **Problem 1**: A simple dungeon with a single locked corridor and basic key retrieval.
2. **Problem 2**: Increased complexity with multiple locks and interconnected paths.
3. **Problem 3**: Challenges include risky corridors and colored locks requiring strategic planning.
4. **Problem 4** *(Original)*: A custom, difficult instance with 7 rooms, requiring over 20 moves and featuring all corridor lock types.

Here is problem 3 as an example:
![](../assets/img/pddl/p2.jpg)

---

## Technical Implementation

### Domain PDDL

The **domain file** defines the structure and rules governing the treasure hunter's world:
- **Predicates**: Represent the hero's location, corridor states (locked, risky, collapsed), room cleanliness, and key properties.
- **Actions**:
  - **Move**: Transition between rooms via unlocked corridors.
  - **Unlock**: Open a corridor using a key of the matching color.
  - **Pickup/Drop Key**: Manage the hero's inventory.
  - **Clean Room**: Remove rubble caused by collapsed corridors.

Key example: The `move` action ensures the corridor is neither locked nor collapsed:

```lisp
(:action move
 :parameters (?from ?to ?corridor)
 :precondition (and (hero-at ?from)
                    (cor-connected ?corridor ?from)
                    (cor-connected ?corridor ?to)
                    (not (cor-locked ?corridor))
                    (not (cor-collapsed ?corridor)))
 :effect (and (not (hero-at ?from))
              (hero-at ?to)))
```

### Problem PDDL Files

Each problem file specifies:
- **Initial State**: Hero’s starting position, corridor states, and key locations.
- **Goal State**: Hero at the treasure room with the treasure corridor unlocked.

The Python script `writer.py` automated the generation of initial states by encoding:
- Room and corridor connections.
- Key properties (type, location, color).
- Corridor locks and their colors.

Example output from `writer.py`:
```lisp
(hero-at loc-1-1)

; Corridor connections
(cor-between loc-1-1 loc-2-1 c1121)
(cor-locked c1121)
(cor-lock-colour c1121 yellow)

; Keys
(key-at key1 loc-1-1)
(key-colour key1 yellow)
(key-one-use key1)
```

---

## Learning Experience

This project marked my first exposure to coding in PDDL, a language specifically designed for AI planning. Initially, understanding the syntax and structure of PDDL felt daunting due to its distinct style and logic-based approach. However, as I delved deeper, I gained an appreciation for its power in representing complex systems and constraints. 

Working on this assignment proved to be an invaluable learning exercise, enhancing my understanding of AI planning methodologies and their practical applications. The process not only strengthened my problem-solving skills but also introduced me to new ways of thinking about system modeling and automation. 

---

## Challenges and Solutions

1. **Dynamic Constraints**: 
   - Managing collapsing corridors and messy rooms required careful action sequencing to avoid unsolvable states.
   - Solution: Explicit predicates (`cor-collapsed`, `room-messy`) to track environmental changes.

2. **Complex Key Constraints**: 
   - Modeling multi-use vs. single-use keys without introducing redundant predicates.
   - Solution: Custom fluents like `key-one-use` and conditional effects streamlined this process.

3. **Scalability**: 
   - Generating problem files manually for large dungeons was error-prone.
   - Solution: `writer.py` ensured consistency and reduced development time.

---

## Results and Impact

The final submission met all project requirements:
1. **Domain Modeling**: Fully captures the dynamics of the Treasure Hunter scenario.
2. **Problem Files**: Demonstrates increasing complexity and strategic planning challenges.
3. **Custom Problem**: Adds unique challenges, aligning with the project’s difficulty constraints.

By automating problem file creation and adhering to clean, readable PDDL standards, this project showcases robust AI planning principles.

---

## Conclusion

The Treasure Hunter PDDL project demonstrates the power of AI planning and modeling in solving real-world-inspired problems. Through careful domain design and automation, I developed a scalable and extensible solution for navigating a complex environment. This experience highlights the importance of strategic thinking in AI and provides a strong foundation for future work in planning systems.
