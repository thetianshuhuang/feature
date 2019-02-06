

from akaze import AkazeMatcher

import cv2
import matplotlib.pyplot as plt
from data import Dataset
from print import *


def test_plot(akaze, img, task, hist=False):

    matches, distances = akaze.match(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), task=task)

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(img)
    plt.scatter(
        [match.scene[0] for match in matches],
        [match.scene[1] for match in matches])
    for i, match in enumerate(matches):
        plt.annotate(str(i), match.scene)

    plt.subplot(122)
    plt.imshow(akaze.target)
    plt.scatter(
        [match.target[0] for match in matches],
        [match.target[1] for match in matches])
    for i, match in enumerate(matches):
        plt.annotate(str(i), match.target)

    if hist:
        plt.figure(2)
        plt.hist([i for i in distances if i < 2], bins=20)

    plt.show()


if __name__ == "__main__":
    import syllabus

    print(putil.join("  ", render("feature", SM, BOLD, BR + BLUE)))

    main = syllabus.Task(
        "feature", desc="AKAZE Feature Matcher Test Engine").start()

    m = AkazeMatcher(
        cv2.imread("../hx/ref.PNG"), task=main, name="HeroX Gate")
    d = Dataset()

    test_plot(m, d.get(), main.subtask())

    main.done(join=True)

    # print("\nTrace:")
    # div.div("-")
    # print(main.json(pretty=True))
