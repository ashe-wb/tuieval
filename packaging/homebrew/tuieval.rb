# Homebrew formula template for a tap (e.g. github.com/ashe-wb/homebrew-tap, file Formula/tuieval.rb).
# Users then install with:  brew install ashe-wb/tap/tuieval
#
# Before the first `brew install`, after publishing a release (see packaging/homebrew/README.md):
#   1. set `url` to the release's sdist on PyPI (or the GitHub tag tarball) and `sha256` to its checksum
#   2. run `brew update-python-resources Formula/tuieval.rb` to fill in the resource blocks
#      (textual, pyyaml and their dependencies)
#   3. `brew install --build-from-source Formula/tuieval.rb && brew test tuieval && brew audit --strict tuieval`
class Tuieval < Formula
  include Language::Python::Virtualenv

  desc "Evaluate local LLMs on your own eval packs, in the terminal"
  homepage "https://github.com/ashe-wb/tuieval"
  url "https://files.pythonhosted.org/packages/source/t/tuieval/tuieval-0.1.0.tar.gz"
  sha256 "REPLACE_WITH_SHA256_OF_THE_SDIST"
  license "MIT"

  depends_on "libyaml"
  depends_on "python@3.12"

  # resource blocks go here (brew update-python-resources fills them in)

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/tuieval --version")
    system bin/"tuieval", "init", testpath/"ws"
    assert_path_exists testpath/"ws/models.toml"
    cd testpath/"ws" do
      system bin/"tuieval", "new-pack", "demo"
      assert_match "ok", shell_output("#{bin}/tuieval selftest")
    end
  end
end
