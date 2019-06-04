<?php
final class ClangTidyLinter extends ArcanistExternalLinter {

  private $flags = array();

  public function getInfoName() {
    return 'ClangTidy';
  }

  public function getInfoURI() {
    return 'https://clang.llvm.org/extra/clang-tidy/';
  }

  public function getInfoDescription() {
    return pht('clang-tidy is a clang-based C++ "linter" tool.');
  }

  public function getLinterName() {
    return 'ClangTidy';
  }

  public function getLinterConfigurationName() {
    return 'clang-tidy';
  }

  public function getDefaultBinary() {
    return 'clang-tidy';
  }

  public function getVersion(){
    return 8;
  }

  protected function getMandatoryFlags() {
    return array(
      '--warnings-as-errors=*',
      '--p=/roadstar',
      '--quiet',
    );
  }

  protected function getDefaultFlags() {
    return $this->flags;
  }

  public function getLinterConfigurationOptions(){
    $options = array(
      'clang-tidy.checks' => array(
        'type' => 'optional string',
        'help' => pht('checks filter.'),
      ),
      'clang-tidy.fix' => array(
        'type' => 'optional bool',
        'help' => pht('apply suggested fixes. Default: false'),
      ),
      'clang-tidy.path' => array(
        'type' => 'optional string',
        'help' => 'Path used to read a compile command database. Deault empty.',
      ),
    );
    return $options + parent::getLinterConfigurationOptions();
  }

  public function setLinterConfigurationValue($key, $value) {
    switch ($key) {
      case 'clang-tidy.checks':
        $this->flags[] = '--checks='.$value;
        return;
      case 'clang-tidy.path':
        $this->flags[] = '--p='.$value;
        return;
      case 'clang-tidy.fix':
        if ($value) {
          $this->flags[] = '--fix ';
        }
        return;
    }
    return parent::setLinterConfigurationValue($key, $value);
  }

  public function getInstallInstructions() {
    return pht('sudo apt install clang-tidy');
  }

  protected function canCustomizeLintSeverities() {
    return false;
  }

  protected function parseLinterOutput($path, $err, $stdout, $stderr) {
    preg_match_all('/(?<path>.*):(?<line>\d+):(?<col>\d+):\s+(?<severity>error|warning):(?<msg>.*)\[(?<name>[^,\r\n]*)(,.*)?\]/',
      $stdout, $matches);
    $messages = array();
    for($i = 0; $i < count($matches[0]); $i++) {
      $message = new ArcanistLintMessage();
      $message->setPath($matches['path'][$i]);
      $message->setCode($matches['path'][$i]);
      $message->setLine($matches['line'][$i]);
      $message->setChar($matches['col'][$i]);
      $message->setSeverity($this->mapSeverity($matches['path'][$i], $matches['severity'][$i], $matches['name'][$i]));
      $message->setDescription($matches['msg'][$i]);
      $message->setName($matches['name'][$i]);
      $messages[] = $message;
    }
    return $messages;
  }

  private function startsWith($haystack, $needle)
  {
     $length = strlen($needle);
     return (substr($haystack, 0, $length) === $needle);
  }


  private function mapSeverity($path, $severity, $name) {
    if(!$this->startsWith($path, "/roadstar") || $this->startsWith($path, "/roadstar/bazel-")) {
      return ArcanistLintSeverity::SEVERITY_DISABLED;
    }
    switch($severity) {
      case 'warning':
        return ArcanistLintSeverity::SEVERITY_WARNING;
      case 'error':
      default:
      if ($name === 'clang-diagnostic-error'){ // SKIP clang-diagnostic-error
        return ArcanistLintSeverity::SEVERITY_DISABLED;
      } else {
        return ArcanistLintSeverity::SEVERITY_ERROR;
      }
    }
  }

}
