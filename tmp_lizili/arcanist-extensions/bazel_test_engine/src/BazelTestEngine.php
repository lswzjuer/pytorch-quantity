<?php
final class BazelTestEngine extends ArcanistUnitTestEngine {
  public function runBazel($projectRoot) {
    $command = $this->getConfigurationManager()->getConfigFromAnySource('unit.target');
    if (empty($command)) {
      return [];
    }
    $command = "bazel test --test_output=errors --color=yes " . $command;
    if(getenv('BAZEL_TEST_JOBS') !== false) {
      $command = $command . " --jobs " . getenv('BAZEL_TEST_JOBS');
    }
    $future = new ExecFuture($command);
    $future->setCWD(Filesystem::resolvePath($projectRoot));
    do {
      list($stdout, $stderr) = $future->read();
      echo $stdout;
      echo $stderr;
      sleep(0.5);
    } while (!$future->isReady());
    list($error, $stdout, $stderr) = $future->resolve();
    $results = $this->parseOutput($stdout);
    if ($error != 0) {
      $result = new ArcanistUnitTestResult();
      $result->setName("Overall tests");
      $result->setResult(ArcanistUnitTestResult::RESULT_FAIL);
      $results[] = $result;
    }
    return $results;
  }
  public function shouldEchoTestResults() {
    return true;
  }
  private function parseOutput($output) {
    $results = array();
    $lines = explode(PHP_EOL, $output);
    foreach($lines as $index => $line) {
      // Check NO STATUS
      preg_match('/^(.*)\s+.*NO\sSTATUS/', $line, $matches);
      if (count($matches) >= 1) {
        $result = new ArcanistUnitTestResult();
        $result->setName(trim($matches[1]));
        $result->setResult(ArcanistUnitTestResult::RESULT_BROKEN);
        $results[] = $result;
        continue;
      }

      preg_match('/^(\S*)\s+.*(FAILED|PASSED|TIMEOUT).*\s+in\s+(.*)s/', $line, $matches);
      if (count($matches) < 3) continue;
      $result = new ArcanistUnitTestResult();
      $result->setName(trim($matches[1]));
      $result->setDuration((float) trim($matches[3]));
      switch (trim($matches[2])) {
      case 'PASSED':
        $result->setResult(ArcanistUnitTestResult::RESULT_PASS);
        break;
      case 'FAILED':
      case 'TIMEOUT':
        $exception_message = trim($lines[$index + 1]);
        $result->setResult(ArcanistUnitTestResult::RESULT_FAIL);
        $result->setUserData(file_get_contents($exception_message));
        break;
      default:
        continue;
      }
      $results[] = $result;
    }
    return $results;
  }

  private function runCustom($projectRoot) {
    $targets = $this->getConfigurationManager()->getConfigFromAnySource('unit.custom_targets');
    if (empty($targets)) {
      return [];
    }
    $results = array();
    foreach($targets as $target) {
      $future = new ExecFuture("arcanist-extensions/bazel_test_engine/custom_scripts/" . $target . ".sh");
      $future->setCWD('/roadstar');
      do {
        list($stdout, $stderr) = $future->read();
        echo $stdout;
        echo $stderr;
        sleep(0.5);
      } while (!$future->isReady());
      list($error, $stdout, $stderr) = $future->resolve();
      $result = new ArcanistUnitTestResult();
      $result->setName($target);
      $result->setResult($error == 0 ? ArcanistUnitTestResult::RESULT_PASS : ArcanistUnitTestResult::RESULT_FAIL);
      $results[] = $result;
    }
    return $results;
  }

  public function run() {
    $projectRoot = $this->getWorkingCopy()->getProjectRoot();
    return array_merge($this->runBazel($projectRoot), $this->runCustom($projectRoot));
  }
}
