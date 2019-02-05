
import collections
Match = collections.namedtuple("Match", ["target", "scene"])


class AkazeMatcher:

    def __init__(
            self, target, ratio=0.8, task=None,
            name="Generic Target", samples=50):

        if task is None:
            task = syllabus.Task()

        self.akaze = cv.AKAZE_create()
        self.matcher = cv.DescriptorMatcher_create(
            cv.DescriptorMatcher_FLANNBASED)

        self.name = name
        self.ratio = ratio
        self.sample_size = samples

        self.target = target
        self.kp, self.desc = akaze.detectAndCompute(self.target, None)

    def match(self, scene, task=None):

        if task is None:
            task = syllabus.Task()

        task.start("AKAZE Keypoint Matcher: searching for " + self.name)
        task.print("Searching scene for target...")
        kp, desc = self.akaze.detectAndCompute(scene, None)
        matches = self.matcher.knnMatch(self.desc, desc, 2)

        task.print("Finished keypoint matcher. Running ratio test...")
        matches = [
            Match(
                target=kp[m.trainIdx],
                scene=self.kp[m.queryIdx])
            for m, n in matches
            if m.distance < self.ratio * n.distance]

        task.print("Distance comparison test...")
        if len(matches) * (len(matches) - 1) / 2 > self.sample_size:
            distances = []
            for _ in range(self.sample_size):
                m1, m2 = random.sample(matches, 2)
                distances.append(euclidean(m1.pt, m2.pt))
