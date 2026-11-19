
# README

Personal repo to run simulations based on the game Balatro.


# Goals

The goal is to give myself practice in code development while applying statistical ideas.

## Implementation Goals
- [ ] Fully replicate how a round of Balatro could go
    - Modifiers will not be included unless they are easy to implement
    - [ ] Execute how multiple hands may be played
    - [ ] Score how many chips the achieved hand has
- [ ] Simulate whether straights or flushes are better
- [ ] Empirically test if my poker hand checkers match the expected probabilities
- [ ] Add wild card support

## Research Goals
Things I want to do but not sure how they could be done at the moment

- [ ] Make a strategy function that optimally plays the game
    - [ ] Implement a likelihood strategy to determine which cards to discard
    - [ ] See if i could sample cards to estimate which cards to discard.
        - Some for of importance sampling?
    - [ ] Make a neural network that learns to play the game?
- [ ] Test if my simulation can resolve whether a wild card should pick 3 of a kind vs 2 pair 
    - See this [video](https://youtu.be/eVnPbA0QUec?t=184) for some more context
