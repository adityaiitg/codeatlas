class Codeatlas < Formula
  include Language::Python::Virtualenv

  desc "Local codebase intelligence engine: AST-driven code IR, knowledge graph, living wiki & hybrid search"
  homepage "https://github.com/adityaiitg/codeatlas"
  url "https://files.pythonhosted.org/packages/25/f5/99d8727fe19ac774b5e1b8db8b7d27e223acf22e23747b4556b38c7e07e1/codeatlas_cli-0.1.0.tar.gz"
  sha256 "7461632cbe7a9b58790fbe50247faf5e64d6786a362630322717e2c9314cad1a"
  license "MIT"

  preserve_rpath

  depends_on "python@3.12"

  def install
    virtualenv_create(libexec, "python3.12")
    system "python3.12", "-m", "pip", "--python=#{libexec}/bin/python", "install", buildpath
    bin.install_symlink libexec/"bin/codeatlas"
  end

  test do
    assert_match "CodeAtlas", shell_output("#{bin}/codeatlas --help")
  end
end
